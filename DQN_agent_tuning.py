import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
import optuna # Import Optuna
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

# --- Imports for detailed evaluation ---
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score # Added f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- Configuration ---
# File Paths (Keep these the same)
DATASET_FILE = 'FINAL_DATASET_v4_generalized.csv'
DETECTOR_MODEL_FILE = 'lightgbm_detector_tuned.txt'
FEATURE_LIST_FILE = 'lightgbm_feature_importance_full.csv' # Needed to get feature names

# --- Optuna & Fixed RL Hyperparameters ---
N_OPTUNA_TRIALS = 10 # Number of hyperparameter combinations to try
NUM_TRAIN_ITERATIONS = 20000 # Keep fixed for each trial (can be tuned later)

# --- Other RL Hyperparameters ---
REPLAY_BUFFER_CAPACITY = 100000
COLLECT_STEPS_PER_ITERATION = 1
BATCH_SIZE = 64
LOG_INTERVAL = 5000 # Log less frequently during tuning
EVAL_INTERVAL = 10000 # Evaluate less frequently during tuning
NUM_EVAL_EPISODES = 5 # Fewer episodes for faster eval during tuning

# --- Detector Model Config ---
CATEGORICAL_FEATURES_FOR_DETECTOR = [
    'Id', 'LogonType', 'AuthenticationPackageName', 'TicketEncryptionType', 'DestinationPort'
]

# --- Part A: The RL Environment (Keep AdResponseEnv class exactly the same) ---
class AdResponseEnv(py_environment.PyEnvironment):
    """ (Keep this class unchanged) """
    def __init__(self, data_df, detector_model, rl_state_features, model_features):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(len(rl_state_features) + 1,), dtype=np.float32, name='observation')
        self._df = data_df.reset_index(drop=True); self._detector_model = detector_model
        self._rl_state_features = rl_state_features; self._model_features = model_features
        self._label_col = 'Label' if 'Label' in data_df.columns else 'label'
        self._episode_ended = False; self._current_row_index = 0; self._total_rows = len(self._df)
        # Suppress init message during tuning: # print(f"Environment initialized with {self._total_rows} events.")

    def action_spec(self): return self._action_spec
    def observation_spec(self): return self._observation_spec

    def _reset(self):
        self._current_row_index = np.random.randint(0, self._total_rows); self._episode_ended = False
        state = self._get_state(); return ts.restart(state)

    def _step(self, action):
        if self._episode_ended: return self._reset()
        current_event = self._df.iloc[self._current_row_index]
        reward = self._calculate_reward(action, current_event)
        self._current_row_index += 1
        if self._current_row_index >= self._total_rows: self._episode_ended = True
        next_state = self._get_state()
        if self._episode_ended: return ts.termination(next_state, reward=reward)
        else: return ts.transition(next_state, reward=reward, discount=0.9)

    def _get_state(self):
        current_index = self._current_row_index % self._total_rows
        event_row = self._df.iloc[current_index]; model_input_df = self._prepare_df_for_model(event_row)
        try: confidence = self._detector_model.predict(model_input_df)[0]
        except Exception as e: print(f"[WARN] Predict failed: {e}"); confidence = 0.0
        rl_feature_series = event_row[self._rl_state_features].copy()
        if 'TicketEncryptionType' in rl_feature_series.index:
            try:
                val_str = str(rl_feature_series['TicketEncryptionType'])
                rl_feature_series['TicketEncryptionType'] = int(val_str, 16) if val_str.startswith('0x') else int(float(val_str))
            except (ValueError, TypeError): rl_feature_series['TicketEncryptionType'] = -1
        if 'Id' in rl_feature_series.index:
            try: rl_feature_series['Id'] = int(float(rl_feature_series['Id']))
            except (ValueError, TypeError): rl_feature_series['Id'] = -1
        rl_feature_values = rl_feature_series.values
        state_vector = np.concatenate(([confidence], rl_feature_values)); return state_vector.astype(np.float32)

    def _prepare_df_for_model(self, event_row):
        model_input_df = pd.DataFrame([event_row[self._model_features]])
        for col in CATEGORICAL_FEATURES_FOR_DETECTOR:
            if col in model_input_df.columns:
                try:
                    is_num_like = pd.api.types.is_numeric_dtype(model_input_df[col].dtype) or \
                                  (model_input_df[col].dtype == 'object' and str(model_input_df[col].iloc[0]).replace('-','').isnumeric())
                    model_input_df[col] = model_input_df[col].fillna(-1).astype(int) if is_num_like else model_input_df[col].fillna('Unknown').astype(str)
                    model_input_df[col] = model_input_df[col].astype('category')
                except Exception: model_input_df[col] = model_input_df[col].fillna(0)
        return model_input_df

    def _calculate_reward(self, action, current_event):
        """ (Keep the reward function unchanged - the final balanced one) """
        action_int = action.item() if isinstance(action, np.ndarray) else action
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
                if action_int <= 1:
                    return -10000.0 # CRITICAL FAILURE
                elif action_int == 2:
                    return -500.0
                elif action_int == 3:
                    return -200.0
                elif action_int == 4:
                    return 10000.0  # BEST RESPONSE
            else:
                # --- CASE 2: NON-CRITICAL MALICIOUS EVENT ---
                if action_int <= 1:
                    return -5000.0 # Failure
                elif action_int == 2:
                    return 5000.0    # BEST RESPONSE
                elif action_int == 3:
                    return 3000.0
                elif action_int == 4:
                    return -500.0   # Overkill

        else: # --- It WAS Benign (true_label == 0) ---
            if action_int == 0:
                return 10.0      # SUCCESS
            elif action_int == 1:
                return 1.0       # Acceptable
            elif action_int == 2:
                return -5000.0   # FP
            elif action_int == 3:
                return -8000.0   # Major FP
            elif action_int == 4:
                return -10000.0  # Crit FP

        return 0.0 # Default

# --- Helper Function for Avg Return Evaluation (Keep compute_avg_return exactly the same) ---
def compute_avg_return(environment, policy, num_episodes=NUM_EVAL_EPISODES): # Use constant
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset(); episode_return = 0.0
        for _ in range(500): # Simulate 500 steps
            if time_step.is_last(): break
            action_step = policy.action(time_step); time_step = environment.step(action_step.action)
            episode_return += time_step.reward.numpy()[0]
        total_return += episode_return
    return total_return / num_episodes

# --- Helper Function for Full Metrics Evaluation (Modified to return F1 score) ---
def evaluate_agent_and_save_metrics(environment, policy, results_dir, model_name, num_eval_steps=5000):
    """ (Mostly unchanged, but returns macro F1 score) """
    print(f"\n--- [Running Full Evaluation for {model_name}] ---")
    y_true, y_action = [], []
    py_env = environment.pyenv.envs[0]; label_col = py_env._label_col
    time_step = environment.reset()
    eval_step_count = 0
    while eval_step_count < num_eval_steps: # Ensure exactly num_eval_steps
        if time_step.is_last(): time_step = environment.reset()
        current_idx = py_env._current_row_index % py_env._total_rows
        try:
             true_label = int(py_env._df.iloc[current_idx][label_col]) # Ensure int
             y_true.append(true_label)
        except IndexError: time_step = environment.reset(); continue # Skip if index issue

        action_step = policy.action(time_step); y_action.append(action_step.action.numpy()[0])
        time_step = environment.step(action_step.action)
        eval_step_count += 1

    y_pred_binary = [1 if a > 1 else 0 for a in y_action]
    acc = accuracy_score(y_true, y_pred_binary)
    # Use output_dict=True to easily extract F1 score
    report_dict = classification_report(y_true, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'], zero_division=0, output_dict=True)
    report_str = classification_report(y_true, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'], zero_division=0)
    macro_f1 = report_dict['macro avg']['f1-score'] # Extract the score to return

    report_header = f"--- Performance Report: {model_name} ---\n"; report_header += f"Evaluated on {num_eval_steps} steps from the test set.\n\n"
    acc_str = f"Overall Accuracy (Active/Passive): {acc:.4f}\n\n"; class_report_str = f"Classification Report (Active/Passive vs. True Label):\n{report_str}\n"
    full_report = report_header + acc_str + class_report_str
    print(full_report)
    report_filename = os.path.join(results_dir, f"{model_name}_classification_report.txt");
    with open(report_filename, 'w') as f: f.write(full_report)
    print(f"-> Text report saved to '{report_filename}'")

    cm_binary = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6)); sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted: Benign (Passive)', 'Predicted: Malicious (Active)'], yticklabels=['True: Benign (0)', 'True: Malicious (1)'])
    plt.title(f'{model_name} - Binary Confusion Matrix', fontsize=16); plt.ylabel('Actual Label'); plt.xlabel('Predicted Action Type')
    cm_binary_filename = os.path.join(results_dir, f"{model_name}_binary_confusion_matrix.png"); plt.savefig(cm_binary_filename); plt.close()
    print(f"-> Binary CM plot saved to '{cm_binary_filename}'")

    cm_detailed_full = confusion_matrix(y_true, y_action, labels=[0, 1, 2, 3, 4])
    # Ensure cm_detailed_full has expected shape before slicing
    if cm_detailed_full.shape[0] >= 2:
        cm_detailed_sliced = cm_detailed_full[0:2, :]
        cm_detailed_df = pd.DataFrame(cm_detailed_sliced, columns=[f'Action: {i}' for i in range(5)], index=['True: Benign (0)', 'True: Malicious (1)'])
        plt.figure(figsize=(10, 5)); sns.heatmap(cm_detailed_df, annot=True, fmt='d', cmap='Oranges')
        plt.title(f'{model_name} - Detailed Action Confusion Matrix', fontsize=16); plt.ylabel('Actual Label'); plt.xlabel('Agent\'s Chosen Action')
        action_names = ['0: Do Nothing', '1: Alert', '2: Kill Process', '3: Disable User', '4: Isolate Host']
        plt.xticks(ticks=np.arange(5) + 0.5, labels=action_names, rotation=45, ha='right')
        cm_detailed_filename = os.path.join(results_dir, f"{model_name}_detailed_action_matrix.png"); plt.savefig(cm_detailed_filename, bbox_inches='tight'); plt.close()
        print(f"-> Detailed Action Matrix plot saved to '{cm_detailed_filename}'")
    else:
        print("[WARN] Could not generate detailed action matrix due to unexpected confusion matrix shape.")

    print("--- Full Evaluation Complete ---")
    return macro_f1 # Return the score for Optuna

# --- Function to Run One Training Trial ---
def run_single_training_trial(trial_num, learning_rate, hidden_units, num_train_iterations):
    """ Runs one full training and evaluation session. Returns the final macro F1 score."""

    run_name = f"trial_{trial_num}"
    policy_save_dir = f'rl_agent_policy_{run_name}'
    results_dir = f'rl_agent_performance_{run_name}'

    print(f"\n===== Starting Trial: {run_name} =====")
    print(f"  Learning Rate: {learning_rate:.6f}")
    print(f"  Hidden Units: {hidden_units}")
    print(f"  Iterations: {num_train_iterations}")
    print("===================================")

    os.makedirs(results_dir, exist_ok=True) # Create results dir for this trial

    # --- Load Data & Models (Should happen only once ideally, but simple here) ---
    try:
        lgb_booster = lgb.Booster(model_file=DETECTOR_MODEL_FILE)
        MODEL_FEATURES = lgb_booster.feature_name()
        RL_STATE_FEATURES_TOP10 = ['targets_sensitive_object', 'Id', 'ProcessName_freq','Role_DomainController','FileActivityVolume_1min', 'SubjectUserName_freq', 'cmd_length','Role_Workstation', 'FailedLogonCount_5min', 'IsLsassAccess']
        RL_STATE_FEATURES_ADDITIONAL = ['cmd_is_hidden', 'ParentProcessName_freq', 'WorkstationName_freq','TargetOutboundUserName_freq', 'is_in_suspicious_dir', 'IsNewService','IsShadowCopyDeletion', 'WeakKerberosRequestCount_10min', 'cmd_is_encoded','TicketEncryptionType', 'Role_Unknown']
        temp_rl_features = list(set(RL_STATE_FEATURES_TOP10 + RL_STATE_FEATURES_ADDITIONAL))
        RL_STATE_FEATURES = [f for f in temp_rl_features if f in MODEL_FEATURES] # Filter based on detector
        label_col = 'Label' if 'Label' in pd.read_csv(DATASET_FILE, nrows=1).columns else 'label'
        cols_to_load = list(set(MODEL_FEATURES + RL_STATE_FEATURES + [label_col]))
        full_df = pd.read_csv(DATASET_FILE, low_memory=False, usecols=cols_to_load)
    except Exception as e: print(f"[FATAL] Trial {trial_num} failed during data loading: {e}"); return 0.0 # Return 0 score on failure

    # --- Setup Environments ---
    train_df, eval_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df[label_col])
    train_py_env = AdResponseEnv(train_df, lgb_booster, RL_STATE_FEATURES, MODEL_FEATURES)
    eval_py_env = AdResponseEnv(eval_df, lgb_booster, RL_STATE_FEATURES, MODEL_FEATURES)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # --- Initialize Agent ---
    num_actions = train_env.action_spec().maximum + 1
    num_state_features = train_env.observation_spec().shape[0]
    q_network_layers = [tf.keras.layers.InputLayer(input_shape=(num_state_features,))]
    for units in hidden_units: q_network_layers.append(tf.keras.layers.Dense(units, activation='relu'))
    q_network_layers.append(tf.keras.layers.Dense(num_actions, activation=None))
    q_net = sequential.Sequential(q_network_layers, name=f'QNetwork_{run_name}')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimizer,
                               td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=train_step_counter)
    agent.initialize()

    # --- Setup Training Components ---
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size, max_length=REPLAY_BUFFER_CAPACITY)
    collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=COLLECT_STEPS_PER_ITERATION)
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=BATCH_SIZE, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # --- Train Agent ---
    print(f"-> Starting training for {num_train_iterations} iterations...")
    agent.train = common.function(agent.train) # Optimize training step
    collect_driver.run(train_env.reset())
    start_time = time.time()
    initial_avg_return = compute_avg_return(eval_env, agent.policy, NUM_EVAL_EPISODES) # Initial eval
    print(f'Iteration 0: Average Return = {initial_avg_return:.2f}')

    for i in range(num_train_iterations):
        collect_driver.run(); experience, _ = next(iterator); train_loss = agent.train(experience)
        step = agent.train_step_counter.numpy()
        if step > 0 and step % LOG_INTERVAL == 0: print(f'   Step {step}: loss = {train_loss.loss:.4f}')
        if step > 0 and step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, NUM_EVAL_EPISODES)
            print(f'Step {step}: Average Return = {avg_return:.2f}')
    end_time = time.time()
    print(f"-> Training complete in {end_time - start_time:.2f} seconds.")

    # --- Final Evaluation ---
    final_macro_f1 = evaluate_agent_and_save_metrics(eval_env, agent.policy, results_dir, f"DQN_{run_name}")

    # --- Save Policy ---
    os.makedirs(policy_save_dir, exist_ok=True)
    saver = policy_saver.PolicySaver(agent.policy); saver.save(policy_save_dir)
    print(f"-> Agent's policy saved to '{policy_save_dir}'")
    print(f"===== Finished Trial: {run_name} | Final Macro F1: {final_macro_f1:.4f} =====")

    # --- Return metric for Optuna ---
    return final_macro_f1

# --- Optuna Objective Function ---
def objective(trial):
    """ Optuna objective function to sample hyperparameters and run a trial. """
    # --- Suggest Hyperparameters ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) # Log scale for LR
    n_units_l1 = trial.suggest_categorical("n_units_l1", [64, 128, 256])
    n_units_l2 = trial.suggest_categorical("n_units_l2", [64, 128, 256])
    hidden_units = (n_units_l1, n_units_l2)
    # num_iterations = trial.suggest_int("iterations", 10000, 50000, step=10000) # Optional

    # --- Run the training trial ---
    # Pass fixed parameters along with suggested ones
    score = run_single_training_trial(
        trial_num=trial.number,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        num_train_iterations=NUM_TRAIN_ITERATIONS, # Using fixed iterations
        # Add other necessary fixed params if run_single_training_trial needs them
        # replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        # batch_size=BATCH_SIZE,
        # log_interval=LOG_INTERVAL,
        # eval_interval=EVAL_INTERVAL,
        # num_eval_episodes=NUM_EVAL_EPISODES
    )

    # --- Return the score Optuna should maximize ---
    return score


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Check files ---
    if not os.path.exists(DATASET_FILE): print(f"[FATAL] Input dataset '{DATASET_FILE}' not found.")
    elif not os.path.exists(DETECTOR_MODEL_FILE): print(f"[FATAL] Detector model '{DETECTOR_MODEL_FILE}' not found.")
    elif not os.path.exists(FEATURE_LIST_FILE): print(f"[FATAL] Feature list '{FEATURE_LIST_FILE}' not found.")
    else:
        # --- Create and run Optuna study ---
        study = optuna.create_study(direction="maximize") # Maximize the F1 score returned by objective
        print(f"\n--- Starting Optuna Study: {N_OPTUNA_TRIALS} Trials ---")
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

        # --- Print Best Results ---
        print("\n--- Optuna Study Complete ---")
        print(f"Number of finished trials: {len(study.trials)}")
        try:
            best_trial = study.best_trial
            print("Best trial:")
            print(f"  Value (Max Macro F1): {best_trial.value:.4f}")
            print("  Best Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            # Recommend next steps
            best_run_name = f"trial_{best_trial.number}"
            print("\nTo use the best model:")
            print(f"1. Find the saved policy in: 'rl_agent_policy_{best_run_name}'")
            print(f"2. Check the performance reports in: 'rl_agent_performance_{best_run_name}'")
            print("3. You may want to retrain this best configuration for more iterations if needed.")
        except ValueError:
            print("Optuna study finished, but no trials were completed successfully.")

        print("\n--- Tuning Process Complete ---")