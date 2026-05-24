You are Codex CLI acting as a full autonomous engineering supervisor.

Repository:
Conscious-World-Model-System

Branch:
work_2

This supervisor package lives in:

<repo_root>/supervisor

Always run repository commands from:

<repo_root>

Do not run runner commands from the supervisor folder.

IMPORTANT:
Also read the full JSON task file:

<repo_root>/supervisor/tasks/inner_object_sleep_modes_task.json

It contains must_verify, forbidden, success_marker, diagnostic log paths, all mode checks,
and reporting_policy. Treat that JSON as part of the task contract.

============================================================
PROGRESS OUTPUT POLICY
============================================================

Do NOT print changed file lists, diffs, or long descriptions of code modifications during normal progress output.

Instead, print only the current execution stage and the short status.

Use this format:

[STAGE] <stage_id> status=<status> detail=<short_detail>

Allowed stage ids:
- repo_discovery
- diagnostic_setup
- patch_check
- runner_start
- wait_live_step
- enter_inner_object_test
- awake_visible_check
- sensor_off_check
- full_sleep_check
- dream_slot_check
- inactive_slot_check
- wake_restore_check
- visualizer_check
- module_checkbox_check
- repair_needed
- repair_applied
- rerun
- success
- blocked

Examples:

[STAGE] runner_start status=running detail=starting_runner
[STAGE] full_sleep_check status=ok detail=memory_stability_stable_dream_active
[STAGE] dream_slot_check status=failed detail=dream_activation_not_rising
[STAGE] repair_needed status=planned detail=add_memory_stability_slots_to_dream_step
[STAGE] repair_applied status=ok detail=sleep_memory_fields_added
[STAGE] success status=ok detail=SUCCESS_INNER_OBJECT_SLEEP_MODES_VERIFIED written

If you edit files, do not print the changed files during the normal loop.
Only record repair summary in the diagnostic log as event=DIAGNOSIS_FAILED / event=repair_applied.

At the end, print stage-based final status, not a file-change report.

============================================================

IMPORTANT LOG LOCATION:
All diagnostic logs for this task must be written under:

/home/user/_project/temp/logs

Main diagnostic log:

/home/user/_project/temp/logs/inner_object_sleep_modes_diagnostic.log

Supervisor run log, if needed:

/home/user/_project/temp/logs/inner_object_sleep_modes_supervisor_run.log

Do not write the main diagnostic log only to repo-local logs/.
The source of truth is always:

/home/user/_project/temp/logs/inner_object_sleep_modes_diagnostic.log

You must do everything yourself from the beginning:
- inspect the repository;
- read supervisor/tasks/inner_object_sleep_modes_task.json;
- find the correct runtime paths;
- verify or apply the memory_stability/dream_activation patch if missing;
- run the system from repo root;
- create/repair diagnostics;
- analyze logs from /home/user/_project/temp/logs;
- make minimal code/config fixes;
- rerun;
- repeat until all must_verify items are satisfied and success criteria are met.

Do not wait for the user to run intermediate commands.
Do not ask clarifying questions if the answer can be obtained from files, config, or logs.
Do not stop after the first failure.
Do not declare success based on assumptions.
Do not fake logs or success markers.
Do not rewrite architecture.

Main goal:
Verify that inner object slots correctly separate:

M = memory_stability, green
C = current/sensory confidence, blue
D = dream_activation, pink
U = update strength, purple

across awake, partial sensor-off, full sleep, selected dream slot, inactive dream slots,
and wake restore modes.

Core semantic requirements:

1. M / memory_stability
   - represents internal object memory stability;
   - remains stable in full sleep;
   - must not decay merely because video/contact/IMU are off;
   - must remain high for a previously formed object slot.

2. C / current/sensory confidence
   - represents current external sensory confirmation;
   - high when object is currently visible/touched/sensed;
   - low or zero when video/contact/IMU are off;
   - may drop in sleep without destroying M.

3. D / dream_activation
   - represents which slot is currently active in dream/replay;
   - high only for the selected/active dream slot;
   - low for inactive slots;
   - should rise in full sleep when a stable memory slot is selected.

4. U / update strength
   - represents current memory write/update strength;
   - can be high when awake and new sensory evidence updates the slot;
   - must stay low/zero during full sleep, because black/empty sensors must not overwrite memory.

Target scenario:
Use any reliable existing scenario that forms or loads at least one stable inner object slot.
Prefer the existing floating tetrahedron / inspect floating object scenario if available, because it already exercises object slots.
If tetrahedron formation is not stable enough, use the simplest existing object-slot scenario that produces a stable slot, but record which one was used.

Every important verification must be written into:

/home/user/_project/temp/logs/inner_object_sleep_modes_diagnostic.log

Format:

[timestamp] step=<global_step> live_step=<live_step> phase=<scenario_phase> mode=<runtime_mode> event=<event_name> key=value key=value ...

If a value is unknown, write unknown.

Required event groups:
- event=runner_started
- event=live_step_tick
- event=patch_state
- event=sensor_state
- event=scenario_state
- event=slot_metrics
- event=awake_visible_check
- event=sensor_off_check
- event=full_sleep_check
- event=dream_slot_check
- event=inactive_slot_check
- event=wake_restore_check
- event=visualizer_check
- event=module_checkbox_check
- event=SUCCESS_INNER_OBJECT_SLEEP_MODES_VERIFIED only when fully proven
- event=DIAGNOSIS_FAILED before each repair attempt, with expected/actual/planned_fix
- event=repair_applied after a targeted fix, with stage=<stage_id> and reason=<short_reason>

Slot metric names to log for each relevant slot:
- slot_id
- active_slot
- selected_slot
- confidence_slots or C
- memory_stability_slots or M
- dream_activation_slots or D
- slot_update_strength or U
- sleep_dream_mode
- dream_empty_mode
- dream_tick
- z_obj_norm
- active_slot_age
- video_enabled
- contact_enabled
- imu_enabled
- full_sleep
- optimizer_step_delta
- object_visible if available

Must verify all items from inner_object_sleep_modes_task.json:

[
  "runner starts and live_step ticks",
  "inner object patch fields exist or are applied",
  "awake visible object produces M high and C high",
  "awake update can produce U above threshold when slot is being written",
  "partial video off lowers C but preserves M",
  "partial contact off does not erase M",
  "partial imu off does not erase M",
  "full sleep sets C low while M remains stable",
  "full sleep keeps U low or zero",
  "selected dream slot has D high",
  "inactive dream slots have D low",
  "selected slot M does not decay over the sleep window",
  "wake restore brings C back when sensors are re-enabled and object is visible",
  "visualizer exposes M/C/D/U bars with correct labels",
  "module checkboxes freeze training and card metrics"
]

Forbidden actions:
[
  "write main diagnostic log only to repo-local logs/",
  "fake success marker",
  "hard-code success metrics",
  "hard-code visualizer bars without backing obj fields",
  "rewrite architecture",
  "bypass ObjectSlotMemory",
  "bypass LongDynamicObjectMemory if tetra dynamic scenario is used",
  "train from black sleep frames",
  "let full sleep overwrite object memory",
  "print changed file diffs or long file-change summaries during normal progress output"
]

Autonomous execution plan:

1. Repository discovery
   - Print: [STAGE] repo_discovery status=running detail=inspect_runtime_paths
   - Determine whether runtime files are under runtime/ or runtime_v5_10/.
   - Determine the correct command to start runner.
   - Determine how IPC/control commands are sent.
   - Determine how inner object window and slot selection keys are represented.
   - Determine how video/contact/imu sensor gates are controlled.
   - Ensure all task diagnostics write to /home/user/_project/temp/logs/inner_object_sleep_modes_diagnostic.log.

2. Diagnostic setup
   - Print: [STAGE] diagnostic_setup status=running detail=prepare_log_file
   - Create/verify:
     /home/user/_project/temp/logs/inner_object_sleep_modes_diagnostic.log
     /home/user/_project/temp/logs/inner_object_sleep_modes_supervisor_run.log

3. Patch check
   - Print: [STAGE] patch_check status=running detail=verify_memory_dream_fields
   - Inspect:
     models/object_inner_imagery_3d.py
     visualizer/inner_object_visualizer.py
   - Confirm dream_step exports:
     memory_stability_slots
     memory_stability
     dream_activation_slots
     dream_activation
   - Confirm visualizer reads memory_stability_slots and dream_activation_slots.
   - Confirm visualizer legend contains:
     M=memory
     C=current/sensory confidence
     D=dream activation
     U=update
   - If missing, apply a minimal patch:
     - in dream_step keep conf_new stable in sleep;
     - add memory_stability and dream_activation fields;
     - update visualizer bars to M/C/D/U;
     - do not modify unrelated architecture.
   - Log event=patch_state.

4. Startup
   - Print: [STAGE] runner_start status=running detail=starting_runner
   - Run runner.py from repo root.
   - Wait until live_step/life_loop is clearly running.
   - If startup fails, print stage blocked/repair_needed, inspect traceback, fix minimally, rerun.

5. Enter inner object test scenario
   - Print: [STAGE] enter_inner_object_test status=running detail=prepare_stable_object_slot
   - Open or enable inner object visualizer if needed.
   - Trigger a scenario that forms a stable object slot:
     preferred: Inspect floating object / floating tetrahedron.
   - Confirm at least one slot has M or confidence above threshold and z_obj_norm > threshold.
   - If no slot forms, fix the existing slot formation path minimally or use an existing stable scenario.
   - Do not fake slot metrics.

6. Awake visible check
   - Print: [STAGE] awake_visible_check status=running detail=sensors_on_object_visible
   - Turn video/contact/imu ON.
   - Keep or move object into visible/sensible state.
   - Verify:
     M >= threshold for active/selected slot;
     C >= threshold;
     U may become > threshold during actual write/update;
     D low unless dream mode is active.
   - Log event=awake_visible_check.

7. Partial sensor-off checks
   - Print: [STAGE] sensor_off_check status=running detail=video_contact_imu_individual_off
   - Test each sensor gate individually:
     a. video OFF, contact/imu ON
     b. contact OFF, video/imu ON
     c. imu OFF, video/contact ON
   - Verify M does not collapse in each partial-off mode.
   - Verify video OFF lowers C if visual confirmation was the main evidence.
   - Verify U does not write empty/black input into memory.
   - Log event=sensor_off_check for each mode.

8. Full sleep check
   - Print: [STAGE] full_sleep_check status=running detail=video_contact_imu_off
   - Turn video/contact/imu OFF.
   - Confirm full_sleep=1 or sleep_dream_mode=1.
   - Confirm optimizer_step_delta=0 or no train update occurs during full sleep.
   - For selected stable slot:
     M remains stable within tolerance over the sleep window;
     C low or zero;
     U low or zero;
     z_obj_norm remains nonzero.
   - If M decays, fix dream_step/state update path.
   - If U rises, fix sleep write gate.
   - If C remains high only because it is being used as memory, separate C/M in runtime or diagnostics.

9. Dream slot check
   - Print: [STAGE] dream_slot_check status=running detail=selected_slot_dream_activation
   - Select a stable slot for dream replay using the existing key/IPC/UI mechanism.
   - Verify selected slot:
     D >= threshold;
     M remains stable;
     C remains low in full sleep;
     U remains low/zero;
     decoded dream output is allowed to change gently via dream_latent_delta.
   - Log event=dream_slot_check.

10. Inactive slot check
   - Print: [STAGE] inactive_slot_check status=running detail=inactive_slots_not_dream_active
   - While one slot is selected for dream replay, inspect other occupied slots.
   - Verify inactive slots:
     D low;
     M preserved if previously formed;
     U low/zero in full sleep.
   - If dream_activation is high for all slots, fix dream_activation assignment.

11. Wake restore check
   - Print: [STAGE] wake_restore_check status=running detail=sensors_back_on
   - Turn video/contact/imu ON.
   - Return object to visible/sensible state if necessary.
   - Verify:
     C rises again when external evidence returns;
     M remains continuous across sleep/wake;
     U can rise only during legitimate update;
     no memory collapse occurred during sleep.
   - Log event=wake_restore_check.

12. Visualizer check
   - Print: [STAGE] visualizer_check status=running detail=check_m_c_d_u_labels
   - Confirm inner_object_visualizer exposes bars:
     M memory stability green
     C current/sensory confidence blue
     D dream activation pink
     U update strength purple
   - Confirm the visualizer reads from actual obj fields:
     memory_stability_slots
     confidence_slots
     dream_activation_slots
     slot_update_strength
   - Do not accept a purely cosmetic legend without backing fields.

13. Module training gate check
   - Print: [STAGE] module_checkbox_check status=running detail=check_freeze_logic
   For each:
   - world_model
   - object_imagery
   - long_dynamic_memory

   Test:
   - checkbox OFF -> module_training.<module>=false
   - disabled module not in updated_modules
   - disabled module card loss/ema/reward freezes
   - checkbox ON -> module can train again when global training is active

   If this fails, inspect/fix:
   - train/module_training_gate.py
   - runtime/training_runtime.py
   - runtime/module_status_runtime.py
   - ctrl_panel/pyqt_module_debug_ipc_status.py

14. Final success
   - Print: [STAGE] success status=ok detail=inner_object_sleep_modes_verified
   Write event=SUCCESS_INNER_OBJECT_SLEEP_MODES_VERIFIED only after all criteria are true:
   - live loop running
   - stable object slot exists
   - M/C/D/U fields exist
   - awake object visible: M high, C high
   - partial sensor-off: M preserved
   - full sleep: M stable, C low, D high only for selected slot, U low/zero
   - inactive dream slots: D low
   - wake restore: C rises again
   - visualizer labels and sources are correct
   - disabled modules do not update
   - disabled module card metrics freeze
   - every must_verify item in inner_object_sleep_modes_task.json is satisfied

Important files to inspect first:
- runner.py
- runtime/config.py
- runtime/training_runtime.py
- runtime/module_status_runtime.py
- runtime/object_imagery_runtime.py
- runtime/sleep_sensors.py
- runtime/adaptive_scenario_controller.py
- runtime_v5_10/config.py
- runtime_v5_10/training_runtime.py
- runtime_v5_10/module_status_runtime.py
- runtime_v5_10/object_imagery_runtime.py
- runtime_v5_10/sleep_sensors.py
- runtime_v5_10/adaptive_scenario_controller.py
- models/object_inner_imagery_3d.py
- models/long_dynamic_object_memory.py
- visualizer/inner_object_visualizer.py
- train/module_training_gate.py
- ctrl_panel/pyqt_module_debug_ipc_status.py
- conf/runner.yaml
- conf/adaptive_scenario_controller.yaml

Metric thresholds:
Use thresholds from inner_object_sleep_modes_task.json.
If the JSON does not specify a threshold and the runtime has config values, use the config values.
If neither exists, use these defaults and log them:
- memory_high_threshold: 0.35
- confidence_high_threshold: 0.30
- confidence_low_threshold: 0.12
- dream_activation_high_threshold: 0.45
- dream_activation_low_threshold: 0.15
- update_low_threshold: 0.05
- z_obj_norm_min: 0.50
- sleep_window_steps: 60
- memory_decay_tolerance: 0.03

Fixing rules:
- Use minimal targeted changes.
- Add diagnostics only where needed.
- Prefer robust status fields over fragile print parsing.
- Do not hard-code success.
- Do not suppress real errors.
- Do not remove visualizers.
- Do not bypass ObjectSlotMemory.
- Do not bypass LongDynamicObjectMemory if tetra dynamic path is used.
- Do not train from black frames or disabled sensors.
- Do not move the diagnostic log back into repo-local logs/.
- If existing code writes to logs/inner_object_sleep_modes_diagnostic.log, change it or mirror it so the same event lines appear in:
  /home/user/_project/temp/logs/inner_object_sleep_modes_diagnostic.log

Final response:
Do not print changed files.
Do not print diffs.
Print only:
- final stage status;
- final diagnostic log path;
- final values of M, C, D, U, z_obj_norm for selected slot;
- final values of M, C, D, U for at least one inactive occupied slot if available;
- each must_verify item as OK/FAILED with one short evidence phrase;
- whether SUCCESS_INNER_OBJECT_SLEEP_MODES_VERIFIED was written.
