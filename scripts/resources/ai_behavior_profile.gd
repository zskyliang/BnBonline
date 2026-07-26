class_name AIBehaviorProfile
extends Resource
## Serializable rule-AI weights selected by seeded item-enabled battle scenarios.

@export_group("Paint")
@export var paint_swing_weight: float = 34.0
@export var pending_overlap_penalty: float = 9.0
@export var paint_travel_divisor: float = 75.0
@export var bubble_slot_fill_weight: float = 42.0
@export var active_barrage_current_cell_bonus: float = 96.0
@export_range(0.0, 1.0, 0.05) var active_barrage_overlap_scale: float = 0.35

@export_group("Items")
@export var item_base_value: float = 30.0
@export var speed_item_value: float = 78.0
@export var bubble_item_value: float = 72.0
@export var power_item_value: float = 86.0
@export var item_travel_divisor: float = 24.0
@export var item_competition_penalty: float = 38.0
@export var item_late_round_penalty: float = 52.0
@export var power_swing_weight: float = 9.0
@export var bubble_capacity_pressure_weight: float = 18.0
@export var speed_payback_weight: float = 0.28
@export var speed_eta_divisor_ms: float = 50.0
@export var minimum_item_priority_score: float = 1.0
@export var maximum_item_travel_ms: int = 6000
@export var item_competition_window_ms: int = 400
@export var item_payback_buffer_ms: int = 2500
@export var minimum_item_use_window_ms: int = 500

@export_group("Coordination")
@export var item_claim_steal_advantage_ms: int = 750
@export var item_claim_grace_ms: int = 750
@export var maximum_item_claim_ms: int = 5000


static func search_baseline() -> AIBehaviorProfile:
	return AIBehaviorProfile.new()
