import unittest
import torch

from NSM.utils import (
    StepLearningRateSchedule,
    get_optimizer,
    adjust_learning_rate,
    rename_optimizer_param_groups,
)


def _build_schedules(model_initial=0.005, latent_initial=1e-4, interval=10, factor=0.5):
    return [
        StepLearningRateSchedule(
            initial=model_initial,
            interval=interval,
            factor=factor,
        ),
        StepLearningRateSchedule(
            initial=latent_initial,
            interval=interval,
            factor=factor,
        ),
    ]


def _group_by_param_identity(optimizer, model, latent_vecs, classification_heads=None):
    model_param_ids = {id(p) for p in model.parameters()}
    latent_param_ids = {id(p) for p in latent_vecs.parameters()}
    cls_param_ids = (
        {id(p) for p in classification_heads.parameters()}
        if classification_heads is not None
        else set()
    )

    found = {}
    for group in optimizer.param_groups:
        group_param_ids = {id(p) for p in group["params"]}

        if group_param_ids & latent_param_ids:
            found["latent"] = group
        elif group_param_ids & model_param_ids:
            found.setdefault("model", []).append(group)
        elif cls_param_ids and group_param_ids & cls_param_ids:
            found["classification_heads"] = group

    return found


class TestLRSchedules(unittest.TestCase):
    def assertAlmostEqualRelative(self, a, b, places=12):
        self.assertAlmostEqual(a, b, places=places)

    def test_adjust_lr_maps_schedules_to_correct_groups(self):
        model = torch.nn.Linear(8, 1)
        latent_vecs = torch.nn.Embedding(4, 8)
        schedules = _build_schedules(model_initial=0.005, latent_initial=1e-4)

        optimizer = get_optimizer(
            model,
            latent_vecs,
            lr_schedules=schedules,
            optimizer="AdamW",
            weight_decay=0.0,
        )

        adjust_learning_rate(schedules, optimizer, epoch=1)
        groups = _group_by_param_identity(optimizer, model, latent_vecs)

        self.assertAlmostEqualRelative(
            groups["model"][0]["lr"],
            schedules[0].get_learning_rate(1),
        )
        self.assertAlmostEqualRelative(
            groups["latent"]["lr"],
            schedules[1].get_learning_rate(1),
        )

    def test_lr_decay_follows_correct_schedule_per_group(self):
        model = torch.nn.Linear(8, 1)
        latent_vecs = torch.nn.Embedding(4, 8)
        schedules = _build_schedules(
            model_initial=0.005,
            latent_initial=1e-4,
            interval=10,
            factor=0.5,
        )

        optimizer = get_optimizer(
            model,
            latent_vecs,
            lr_schedules=schedules,
            optimizer="AdamW",
            weight_decay=0.0,
        )

        for epoch in [1, 10, 11, 20, 25]:
            adjust_learning_rate(schedules, optimizer, epoch=epoch)
            groups = _group_by_param_identity(optimizer, model, latent_vecs)

            self.assertAlmostEqualRelative(
                groups["model"][0]["lr"],
                schedules[0].get_learning_rate(epoch),
            )
            self.assertAlmostEqualRelative(
                groups["latent"]["lr"],
                schedules[1].get_learning_rate(epoch),
            )

    def test_resume_round_trip_preserves_mapping(self):
        model = torch.nn.Linear(8, 1)
        latent_vecs = torch.nn.Embedding(4, 8)
        schedules = _build_schedules(model_initial=0.005, latent_initial=1e-4)

        optimizer = get_optimizer(
            model,
            latent_vecs,
            lr_schedules=schedules,
            optimizer="AdamW",
            weight_decay=0.0,
        )

        saved_state = optimizer.state_dict()

        fresh_model = torch.nn.Linear(8, 1)
        fresh_latent_vecs = torch.nn.Embedding(4, 8)
        fresh_optimizer = get_optimizer(
            fresh_model,
            fresh_latent_vecs,
            lr_schedules=schedules,
            optimizer="AdamW",
            weight_decay=0.0,
        )

        fresh_optimizer.load_state_dict(saved_state)
        rename_optimizer_param_groups(
            fresh_optimizer,
            n_model_groups=1,
            has_classification_heads=False,
        )
        adjust_learning_rate(schedules, fresh_optimizer, epoch=1)

        groups = _group_by_param_identity(fresh_optimizer, fresh_model, fresh_latent_vecs)

        self.assertAlmostEqualRelative(
            groups["model"][0]["lr"],
            schedules[0].get_learning_rate(1),
        )
        self.assertAlmostEqualRelative(
            groups["latent"]["lr"],
            schedules[1].get_learning_rate(1),
        )

    def test_adjust_lr_raises_without_group_names(self):
        model = torch.nn.Linear(8, 1)
        latent_vecs = torch.nn.Embedding(4, 8)
        schedules = _build_schedules()

        optimizer = get_optimizer(
            model,
            latent_vecs,
            lr_schedules=schedules,
            optimizer="AdamW",
            weight_decay=0.0,
        )

        for group in optimizer.param_groups:
            group.pop("name", None)

        with self.assertRaisesRegex(KeyError, "missing a recognized 'name'"):
            adjust_learning_rate(schedules, optimizer, epoch=1)

    def test_multi_model_list_all_models_get_schedule0(self):
        model_a = torch.nn.Linear(8, 4)
        model_b = torch.nn.Linear(8, 1)
        latent_vecs = torch.nn.Embedding(4, 8)
        schedules = _build_schedules(model_initial=0.005, latent_initial=1e-4)

        optimizer = get_optimizer(
            [model_a, model_b],
            latent_vecs,
            lr_schedules=schedules,
            optimizer="AdamW",
            weight_decay=0.0,
        )

        adjust_learning_rate(schedules, optimizer, epoch=1)

        model_a_ids = {id(p) for p in model_a.parameters()}
        model_b_ids = {id(p) for p in model_b.parameters()}
        latent_ids = {id(p) for p in latent_vecs.parameters()}

        found_model_lrs = []
        latent_lr = None

        for group in optimizer.param_groups:
            param_ids = {id(p) for p in group["params"]}
            if param_ids & latent_ids:
                latent_lr = group["lr"]
            elif param_ids & model_a_ids:
                found_model_lrs.append(group["lr"])
            elif param_ids & model_b_ids:
                found_model_lrs.append(group["lr"])

        self.assertEqual(len(found_model_lrs), 2)
        expected_model_lr = schedules[0].get_learning_rate(1)
        for lr in found_model_lrs:
            self.assertAlmostEqualRelative(lr, expected_model_lr)

        self.assertAlmostEqualRelative(
            latent_lr,
            schedules[1].get_learning_rate(1),
        )

    def test_too_few_schedules_raises(self):
        model = torch.nn.Linear(8, 1)
        latent_vecs = torch.nn.Embedding(4, 8)
        schedules = [
            StepLearningRateSchedule(
                initial=0.005,
                interval=10,
                factor=0.5,
            )
        ]

        optimizer = get_optimizer(
            model,
            latent_vecs,
            lr_schedules=_build_schedules(),
            optimizer="AdamW",
            weight_decay=0.0,
        )

        with self.assertRaisesRegex(ValueError, "Expected at least 2 lr_schedules"):
            adjust_learning_rate(schedules, optimizer, epoch=1)


if __name__ == "__main__":
    unittest.main()