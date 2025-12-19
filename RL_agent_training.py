import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import py_environment, tf_py_environment, utils
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

# --- [NEW] Imports for detailed evaluation ---
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Added this import

# --- Configuration ---
# File Paths
DATASET_FILE = 'FINAL_DATASET_v4_generalized.csv' # Use the generalized dataset
DETECTOR_MODEL_FILE = 'lightgbm_detector_tuned.txt' # Use the model retrained on generalized data
POLICY_SAVE_DIR = 'rl_agent_policy_expanded' # Save policy in a new directory
RESULTS_DIR = 'rl_agent_performance' # [NEW] Directory to save performance reports

# RL Hyperparameters
NUM_TRAIN_ITERATIONS = 20000  # Total training steps.
REPLAY_BUFFER_CAPACITY = 100000 # Agent's memory size
COLLECT_STEPS_PER_ITERATION = 1 # Collect 1 new log event per training step
BATCH_SIZE = 64               # How many past experiences to learn from at once
LEARNING_RATE = 1e-3          # How fast the neural network learns
LOG_INTERVAL = 1000           # How often to print training loss
EVAL_INTERVAL = 2000          # How often to check the agent's performance
NUM_EVAL_EPISODES = 10        # How many "games" to play to calculate average score

# These are the 5 categorical features your detector model expects
CATEGORICAL_FEATURES_FOR_DETECTOR = [
    'Id', 'LogonType', 'AuthenticationPackageName', 'TicketEncryptionType', 'DestinationPort'
]

# --- Part A: The RL Environment ---
class AdResponseEnv(py_environment.PyEnvironment):
    """
    A custom Python environment for TF-Agents.
    It simulates reading AD logs, gets a prediction from the
    detector model, and asks the RL agent to choose a response.
    """
    def __init__(self, data_df, detector_model, rl_state_features, model_features):
        super().__init__()
        
        # 1. Define Action Space (from your Parameters List)
        # Action 0: Do Nothing
        # Action 1: Alert Only
        # Action 2: Kill Process
        # Action 3: Disable User Account
        # Action 4: Isolate Host
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action'
        )
        
        # 2. Define State Space (based on your feature importance)
        # 1 (Confidence Score) + number of top features
        # This will now be 1 + 21 = 22
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(rl_state_features) + 1,), dtype=np.float32, name='observation'
        )
        
        # 3. Load Data & Model
        self._df = data_df.reset_index(drop=True) # Ensure clean index
        self._detector_model = detector_model
        self._rl_state_features = rl_state_features # Top 21 features for RL state
        self._model_features = model_features       # All 32 features for ML model
        self._label_col = 'Label' if 'Label' in data_df.columns else 'label'
        
        self._episode_ended = False
        self._current_row_index = 0
        self._total_rows = len(self._df)
        print(f"Environment initialized with {self._total_rows} events.")

    # --- TF-Agents Required Methods ---
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # Reset to a random row in the dataset
        self._current_row_index = np.random.randint(0, self._total_rows)
        self._episode_ended = False
        
        state = self._get_state()
        # ts.restart() signals the start of a new episode
        return ts.restart(state)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()
        
        # 1. Get current event
        current_event = self._df.iloc[self._current_row_index]
        true_label = current_event[self._label_col]
        
        # 2. Calculate Reward based on action and true label
        reward = self._calculate_reward(action, current_event) # Pass full event
        
        # 3. Move to the next event
        self._current_row_index += 1
        if self._current_row_index >= self._total_rows:
            self._episode_ended = True
            
        # 4. Get the *next* state
        next_state = self._get_state()
        
        if self._episode_ended:
            # Last step of the episode
            return ts.termination(next_state, reward=reward)
        else:
            # Regular step
            return ts.transition(next_state, reward=reward, discount=0.9) # discount=0.9 values future rewards

    # --- Helper Functions ---
    def _get_state(self):
        """
        Retrieves the state vector for the current log event.
        The state is: [Model_Confidence, Top_N_Features...]
        """
        # Use modulo to wrap around dataset if index goes out of bounds
        current_index = self._current_row_index % self._total_rows
        event_row = self._df.iloc[current_index]
        
        # 1. Get the 32 features the *detector model* needs
        model_input_df = self._prepare_df_for_model(event_row)
        
        # 2. Get Model Confidence (State 1)
        # Use predict(), which returns the probability of class 1
        try:
            confidence = self._detector_model.predict(model_input_df)[0]
        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}")
            confidence = 0.0
        
        # 3. Get the Top N Features for the RL State (S2, S3, ...)
        # --- [FIXED CODE BLOCK] ---
        # We must manually convert any non-numeric features in our RL state
        # before passing them to the neural network.
        
        # Get a *copy* of the features to modify
        rl_feature_series = event_row[self._rl_state_features].copy()

        # Convert 'TicketEncryptionType' (e.g., '0x12') to integer (e.g., 18)
        if 'TicketEncryptionType' in rl_feature_series.index:
            try:
                val_str = str(rl_feature_series['TicketEncryptionType'])
                if val_str.startswith('0x'):
                    rl_feature_series['TicketEncryptionType'] = int(val_str, 16) # Convert from hex
                else:
                    rl_feature_series['TicketEncryptionType'] = int(float(val_str)) # Convert from string/float
            except (ValueError, TypeError):
                rl_feature_series['TicketEncryptionType'] = -1 # Handle NaN or 'Unknown'

        # Convert 'Id' (e.g., '4769') to integer
        if 'Id' in rl_feature_series.index:
            try:
                rl_feature_series['Id'] = int(float(rl_feature_series['Id']))
            except (ValueError, TypeError):
                rl_feature_series['Id'] = -1 # Handle NaN or 'Unknown'
        
        # Now get the all-numeric values
        rl_feature_values = rl_feature_series.values
        # --- [END OF FIX] ---
        
        # 4. Combine into the final state vector
        state_vector = np.concatenate(([confidence], rl_feature_values))
        
        # This will now succeed
        return state_vector.astype(np.float32)

    def _prepare_df_for_model(self, event_row):
        """
        Prepares a single row of data to be 100% compatible with the
        LightGBM model, including correct dtypes.
        """
        # Select the 32 features the model expects
        model_input_df = pd.DataFrame([event_row[self._model_features]])
        
        # Re-apply the categorical dtypes
        for col in CATEGORICAL_FEATURES_FOR_DETECTOR:
            if col in model_input_df.columns:
                try:
                    # Check if the column is numeric-like (e.g., Id, LogonType)
                    if pd.api.types.is_numeric_dtype(model_input_df[col].dtype) or (model_input_df[col].dtype == 'object' and str(model_input_df[col].iloc[0]).replace('-','').isnumeric()):
                        model_input_df[col] = model_input_df[col].fillna(-1).astype(int)
                    else: # Object/String types
                        model_input_df[col] = model_input_df[col].fillna('Unknown').astype(str)
                    
                    # Convert to category
                    model_input_df[col] = model_input_df[col].astype('category')
                except Exception as e:
                    # Fallback
                    model_input_df[col] = model_input_df[col].fillna(0)

        return model_input_df


    def _calculate_reward(self, action, current_event):
        """
        Implements the CONTEXT-AWARE reward function.
        Final tuning: Make FP penalties equal to TP rewards to
        force the agent to be highly confident.
        """
        action_int = action
        if isinstance(action, np.ndarray):
            action_int = action.item() # Get integer value
        
        true_label = int(current_event[self._label_col])

        if true_label == 1: # --- It WAS Malicious ---
            
            is_critical = (
                current_event['FileActivityVolume_1min'] > 200 or
                current_event['targets_sensitive_object'] == 1 or
                current_event.get('IsShadowCopyDeletion', 0) == 1 or
                current_event.get('IsLsassAccess', 0) == 1
            )
            
            if is_critical:
                # --- CASE 1: CRITICAL MALICIOUS EVENT ---
                if action_int == 0 or action_int == 1: return -10000.0 # CRITICAL FAILURE
                elif action_int == 2: return -500.0
                elif action_int == 3: return -200.0
                elif action_int == 4: return 10000.0  # BEST RESPONSE (Keep)
            else:
                # --- CASE 2: NON-CRITICAL MALICIOUS EVENT ---
                if action_int == 0 or action_int == 1: return -5000.0 # Failure
                elif action_int == 2: return 5000.0    # BEST RESPONSE (Keep)
                elif action_int == 3: return 3000.0
                elif action_int == 4: return -500.0
                
        else: # --- It WAS Benign (true_label == 0) ---
            if action_int == 0: return 10.0      # SUCCESS
            elif action_int == 1: return 1.0       # Acceptable
            
            # --- [FIX] Make FP penalties equal to or greater than TP rewards ---
            elif action_int == 2: return -5000.0    # False Positive (Matches non-critical TP)
            elif action_int == 3: return -8000.0    # Major False Positive
            elif action_int == 4: return -10000.0   # CRITICAL FALSE POSITIVE (Matches critical TP)
        
        return 0.0 # Default

# --- Helper Function for Evaluation ---

def compute_avg_return(environment, policy, num_episodes=10):
    """Calculates the average return of a policy over num_episodes."""
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        # Simulate an episode for 500 steps (log events)
        for _ in range(500):
            if time_step.is_last():
                break
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            
            # --- [FIXED LINE] ---
            # Convert the reward Tensor (e.g., [10.0]) to a scalar float (e.g., 10.0)
            episode_return += time_step.reward.numpy()[0]

        total_return += episode_return
    return total_return / num_episodes

# --- [NEW] Helper Function for Full Metrics Evaluation ---
def evaluate_agent_and_save_metrics(environment, policy, results_dir, model_name, num_eval_steps=5000):
    """
    Runs the agent on the environment for N steps and generates a
    classification report and confusion matrix.
    """
    print(f"\n--- [Running Full Evaluation for {model_name}] ---")
    
    y_true = []
    y_action = []
    
    # Get the underlying Python environment to access its properties
    py_env = environment.pyenv.envs[0]
    label_col = py_env._label_col
    
    time_step = environment.reset()
    
    for _ in range(num_eval_steps):
        if time_step.is_last():
            time_step = environment.reset()
        
        # Get the current, true label *before* taking a step
        current_idx = py_env._current_row_index
        true_label = py_env._df.iloc[current_idx][label_col]
        y_true.append(true_label)
        
        # Agent takes an action (this is pure exploitation)
        action_step = policy.action(time_step)
        y_action.append(action_step.action.numpy()[0])
        
        # Environment takes the step
        time_step = environment.step(action_step.action)

    # --- Generate Binary Classification Report ---
    # Map actions (0-4) to binary predictions (0 or 1)
    # Passive actions (0, 1) -> Predict 0 (Benign)
    # Active actions (2, 3, 4) -> Predict 1 (Malicious)
    y_pred_binary = [1 if a > 1 else 0 for a in y_action]
    
    acc = accuracy_score(y_true, y_pred_binary)
    report = classification_report(y_true, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'])
    
    # Build, Print, and Save Text Report
    report_header = f"--- Performance Report: {model_name} ---\n"
    report_header += "Evaluated on {num_eval_steps} steps from the test set.\n\n"
    acc_str = f"Overall Accuracy (Active/Passive): {acc:.4f}\n\n"
    class_report_str = f"Classification Report (Active/Passive vs. True Label):\n{report}\n"
    
    full_report = report_header + acc_str + class_report_str
    print(full_report)
    
    report_filename = os.path.join(results_dir, f"{model_name}_classification_report.txt")
    with open(report_filename, 'w') as f:
        f.write(full_report)
    print(f"-> Text report saved to '{report_filename}'")

    # --- Generate 2x2 Binary Confusion Matrix ---
    cm_binary = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted: Benign (Passive)', 'Predicted: Malicious (Active)'], 
                yticklabels=['True: Benign (0)', 'True: Malicious (1)'])
    plt.title(f'{model_name} - Binary Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Action Type')
    cm_binary_filename = os.path.join(results_dir, f"{model_name}_binary_confusion_matrix.png")
    plt.savefig(cm_binary_filename)
    plt.close()
    print(f"-> Binary CM plot saved to '{cm_binary_filename}'")

    # --- Generate 5x2 Detailed Action Confusion Matrix ---
    # This is the most useful matrix
    
    # This creates a full 5x5 matrix (True Labels 0-4 vs. Predicted Actions 0-4)
    cm_detailed_full = confusion_matrix(y_true, y_action, labels=[0, 1, 2, 3, 4])
    
    # --- [FIX] ---
    # We only care about the first 2 rows (True: 0, True: 1), so we slice it.
    cm_detailed_sliced = cm_detailed_full[0:2, :]
    # --- [END FIX] ---

    cm_detailed_df = pd.DataFrame(cm_detailed_sliced, # Use the new 2x5 sliced matrix
                                  columns=[f'Action: {i}' for i in range(5)], 
                                  index=['True: Benign (0)', 'True: Malicious (1)'])
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm_detailed_df, annot=True, fmt='d', cmap='Oranges')
    plt.title(f'{model_name} - Detailed Action Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Agent\'s Chosen Action')
    # Action labels
    action_names = ['0: Do Nothing', '1: Alert', '2: Kill Process', '3: Disable User', '4: Isolate Host']
    plt.xticks(ticks=np.arange(5) + 0.5, labels=action_names, rotation=45, ha='right')
    
    cm_detailed_filename = os.path.join(results_dir, f"{model_name}_detailed_action_matrix.png")
    plt.savefig(cm_detailed_filename, bbox_inches='tight')
    plt.close()
    print(f"-> Detailed Action Matrix plot saved to '{cm_detailed_filename}'")
    print("--- Full Evaluation Complete ---")


# --- Main Training Function ---
def main_training_loop():
    print("--- [1. Loading and Preparing Environment] ---")
    
    # --- [NEW] Create results directory ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[INFO] Performance reports will be saved to '{RESULTS_DIR}'")
    
    # --- *** MODIFICATION: Define the 21-feature state *** ---
    print("Defining expanded 21-feature state space...")
    # Top 10 from your list
    RL_STATE_FEATURES_TOP10 = [
        'targets_sensitive_object', 'Id', 'ProcessName_freq',
        'Role_DomainController', 'FileActivityVolume_1min', 'SubjectUserName_freq',
        'cmd_length', 'Role_Workstation', 'FailedLogonCount_5min', 'IsLsassAccess'
    ]
    # Your 11 requested additions
    RL_STATE_FEATURES_ADDITIONAL = [
        'cmd_is_hidden', 'ParentProcessName_freq', 'WorkstationName_freq',
        'TargetOutboundUserName_freq', 'is_in_suspicious_dir', 'IsNewService',
        'IsShadowCopyDeletion', 'WeakKerberosRequestCount_10min', 'cmd_is_encoded',
        'TicketEncryptionType', 'Role_Unknown'
    ]
    # Combine and ensure uniqueness
    RL_STATE_FEATURES = list(set(RL_STATE_FEATURES_TOP10 + RL_STATE_FEATURES_ADDITIONAL))
    print(f"Selected {len(RL_STATE_FEATURES)} RL State Features: {RL_STATE_FEATURES}")
    # --- *** END MODIFICATION *** ---
         
    # Load the detection model
    try:
        lgb_booster = lgb.Booster(model_file=DETECTOR_MODEL_FILE)
        MODEL_FEATURES = lgb_booster.feature_name() # Get the 32 features
        print(f"Detector model '{DETECTOR_MODEL_FILE}' loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load detector model '{DETECTOR_MODEL_FILE}': {e}")
        return

    # Load the full dataset (only the columns we need)
    try:
        label_col = 'Label' if 'Label' in pd.read_csv(DATASET_FILE, nrows=1).columns else 'label'
        # Load all features the model needs + the RL state features + the label
        cols_to_load = list(set(MODEL_FEATURES + RL_STATE_FEATURES + [label_col]))
        full_df = pd.read_csv(DATASET_FILE, low_memory=False, usecols=cols_to_load)
        print(f"Full dataset '{DATASET_FILE}' loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset '{DATASET_FILE}': {e}")
        print("Make sure all columns in the feature lists exist in the CSV.")
        return

    # Split data for train and eval environments (80/20 split)
    train_df, eval_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df[label_col])
    
    # Create the TF-Agents environments
    print("Initializing TF-Agents environments...")
    train_py_env = AdResponseEnv(train_df, lgb_booster, RL_STATE_FEATURES, MODEL_FEATURES)
    eval_py_env = AdResponseEnv(eval_df, lgb_booster, RL_STATE_FEATURES, MODEL_FEATURES)
    
    # Wrap them in TF-Agents TF environments
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    print("-> Environments ready.")

    # --- [2. Initializing DQN Agent] ---
    print("\n--- [2. Initializing DQN Agent] ---")
    
    num_actions = train_env.action_spec().maximum + 1
    # State shape comes from the observation spec
    num_state_features = train_env.observation_spec().shape[0]
    print(f"State features: {num_state_features} (1 confidence + {len(RL_STATE_FEATURES)} features), Actions: {num_actions}")
    
    # Define the Q-Network (the "brain")
    # A simple 2-layer neural network
    q_net = sequential.Sequential([
        tf.keras.layers.InputLayer(input_shape=(num_state_features,)), # This automatically uses the new shape (e.g., 22)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation=None) # Q-value for each action
    ], name='QNetwork')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_step_counter = tf.Variable(0)

    # Create the DQN Agent
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )
    agent.initialize()
    print("-> DQN Agent Initialized.")

    # --- [3. Setting up Training Components] ---
    print("\n--- [3. Setting up Training Components] ---")
    
    # The Replay Buffer stores past experiences (State, Action, Reward, Next_State)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=REPLAY_BUFFER_CAPACITY
    )

    # The driver collects experiences from the environment using the agent's policy
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=COLLECT_STEPS_PER_ITERATION # Collect 1 step at a time
    )

    # Create the dataset pipeline for efficient training
    # This samples batches from the replay buffer
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=BATCH_SIZE,
        num_steps=2 # We need 2 steps (s, a, r, s') to train
    ).prefetch(3)
    iterator = iter(dataset)

    print("-> Replay Buffer and Data Collector are ready.")
    
    # --- [4. Training the Agent] ---
    print("\n--- [4. Starting Agent Training] ---")
    print(f"Running {NUM_TRAIN_ITERATIONS} iterations... This will take several minutes.")

    # Optimize the training step with tf.function
    agent.train = common.function(agent.train)
    
    # Reset driver and environment
    collect_driver.run(train_env.reset())
    
    # Initial evaluation
    avg_return = compute_avg_return(eval_env, agent.policy, NUM_EVAL_EPISODES)
    print(f'Iteration 0: Average Return = {avg_return:.2f}')

    for i in range(NUM_TRAIN_ITERATIONS):
        # 1. Collect one step of experience in the environment
        time_step, _ = collect_driver.run()
        
        # 2. Sample a batch of experiences from the replay buffer
        experience, _ = next(iterator)
        
        # 3. Train the agent on that batch
        train_loss = agent.train(experience)
        
        step = agent.train_step_counter.numpy()
        
        if step % LOG_INTERVAL == 0:
            print(f'   Step {step}: loss = {train_loss.loss:.4f}')
        
        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, NUM_EVAL_EPISODES)
            print(f'Step {step}: Average Return = {avg_return:.2f}')
            
    print("-> Training complete.")

    # --- [NEW] 5. Final Full Evaluation ---
    print("\n--- [5. Running Final Evaluation on Test Set] ---")
    # We use agent.policy (exploitation) not agent.collect_policy (exploration)
    evaluate_agent_and_save_metrics(eval_env, agent.policy, RESULTS_DIR, "DQN_Final_Policy")

    # --- [6. Saving the Final Policy] ---
    print("\n--- [6. Saving Final Policy] ---")
    if not os.path.exists(POLICY_SAVE_DIR):
        os.makedirs(POLICY_SAVE_DIR)
        
    # --- [FIX] Use a different variable name to avoid a name conflict ---
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save(POLICY_SAVE_DIR)
    # --- [END FIX] ---

    print(f"-> Agent's policy (the 'brain') saved to '{POLICY_SAVE_DIR}'")
    print("\n--- Process Complete ---")


if __name__ == '__main__':
    # Set the input file to your final, clean, GENERALIZED dataset
    DATASET_FILE = 'FINAL_DATASET_v4_generalized.csv' 
    
    # Set the detector model file (retrained on generalized data)
    # Make sure you have re-run the tuning script on v4 data
    DETECTOR_MODEL_FILE = 'lightgbm_detector_tuned.txt' 
    
    # Set the feature list file (this is just used to get feature names)
    # Make sure you have re-run the feature importance script on the retrained model
    FEATURE_LIST_FILE = 'lightgbm_feature_importance_full.csv' # This file must be regenerated

    # Check if all files exist before starting
    if not os.path.exists(DATASET_FILE):
        print(f"[FATAL] Input dataset '{DATASET_FILE}' not found. Please run 'generalize_hostnames.py' first.")
    elif not os.path.exists(DETECTOR_MODEL_FILE):
        print(f"[FATAL] Detector model '{DETECTOR_MODEL_FILE}' not found. Please re-run the tuning script on the '{DATASET_FILE}' first.")
    elif not os.path.exists(FEATURE_LIST_FILE):
        print(f"[FATAL] Feature importance file '{FEATURE_LIST_FILE}' not found. Please run the feature importance script on the retrained model first.")
    else:
        main_training_loop()