import unittest
from steer.contrastive_pairs import create_pairs


def make_dialogue(values=None, response="Try the north.", span_value="north"):
    values = ["north"] if values is None else values
    state = {"state": [{"slots_values": {
        "slots_values_name": ["hotel-area"], "slots_values_list": [values]
    }}]}
    annotations = {"span_info": {
        "act_type": ["Hotel-Inform"], "act_slot_name": ["area"],
        "act_slot_value": [span_value], "span_start": [8], "span_end": [13]
    }}
    return {"dialogue_id": "test", "turns": {
        "speaker": [0, 1], "turn_id": ["0", "1"],
        "utterance": ["In the north please.", response],
        "frames": [state, {"state": []}],
        "dialogue_acts": [annotations, annotations],
    }}


class ContrastivePairTests(unittest.TestCase):
    def test_system_only_and_shared_history(self):
        pairs = list(create_pairs(make_dialogue()))
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual(p["turn_id"], "1")
        self.assertEqual(p["prompt"], "User: In the north please.\nAssistant:")
        self.assertEqual(p["positive_response"], "Try the north.")
        self.assertNotEqual(p["positive_response"], p["negative_response"])

    def test_excludes_all_allowed_values(self):
        p = list(create_pairs(make_dialogue(["north", "south"])))[0]
        self.assertNotIn(p["negative_value"], ["north", "south"])

    def test_skips_unconstrained_and_mismatched_spans(self):
        for example in [make_dialogue([]), make_dialogue(["dontcare"]), make_dialogue(["south"]),
                        make_dialogue(response="Try the south."),
                        make_dialogue(["north", "east", "south", "west", "centre"])]:
            self.assertEqual(list(create_pairs(example)), [])

    def test_cleared_state_is_not_reused(self):
        example = make_dialogue()
        turns = example["turns"]
        for key in turns:
            turns[key] += turns[key][:]
        turns["frames"][2] = {"state": []}
        self.assertEqual(len(list(create_pairs(example))), 1)

    def test_seed_is_reproducible(self):
        self.assertEqual(list(create_pairs(make_dialogue())), list(create_pairs(make_dialogue())))
