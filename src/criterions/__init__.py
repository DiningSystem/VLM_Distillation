from src.criterions.ce_only import CEOnlyCriterion
from src.criterions.cgkd import CGKDCriterion
from src.criterions.default_distillation import (
    DefaultDistillationCriterion,
    default_distillation_criterion,
)
from src.criterions.dwa_kd import DWAKDCriterion
from src.criterions.dskd_v2 import DSKDv2Criterion
from src.criterions.mcw_kd import MCWKDCriterion

from src.criterions.em_kd import EMKDCriterion
from src.criterions.scva import SCVACriterion
from src.criterions.scva_cgkd import SCVACGKDCriterion
from src.criterions.scva_sic import SCVASICCriterion
from src.criterions.sre import SRECriterion
from src.criterions.unit_aligned import UnitAlignedDistillationCriterion
from src.criterions.ugrimd import UGRIMDCriterion


criterion_list = {
    "ce_only": CEOnlyCriterion,
    "default": DefaultDistillationCriterion,
    "default_distillation": DefaultDistillationCriterion,
    "emkd": EMKDCriterion,
    "em_kd": EMKDCriterion,
    "sre": SRECriterion,
    "dwa_kd": DWAKDCriterion,
    "dwakd": DWAKDCriterion,
    "dskd_v2": DSKDv2Criterion,
    "dskdv2": DSKDv2Criterion,
    "mcw_kd": MCWKDCriterion,
    "mcwkd": MCWKDCriterion,
    "joint": UnitAlignedDistillationCriterion,
    "unit_aligned": UnitAlignedDistillationCriterion,
    "unit_aligned_distillation": UnitAlignedDistillationCriterion,
    "scva": SCVACriterion,
    "cgkd": CGKDCriterion,
    "scva_cgkd": SCVACGKDCriterion,
    "draft": SCVACGKDCriterion,
    "scva_sic": SCVASICCriterion,
    "sic": SCVASICCriterion,
    "ugrimd": UGRIMDCriterion,
    "uncertainty_gated_ranking_interaction": UGRIMDCriterion,
}


def build_criterion(training_args):
    kd_loss_type = getattr(training_args, "kd_loss_type", None)
    if kd_loss_type in (None, ""):
        kd_loss_type = "ce_only"

    if kd_loss_type in criterion_list:
        return criterion_list[kd_loss_type](training_args)

    raise ValueError(f"Unsupported kd_loss_type: {kd_loss_type}")


__all__ = [
    "DefaultDistillationCriterion",
    "DWAKDCriterion",
    "DSKDv2Criterion",
    "MCWKDCriterion",
    "CEOnlyCriterion",
    "EMKDCriterion",
    "SCVACriterion",
    "CGKDCriterion",
    "SCVACGKDCriterion",
    "SCVASICCriterion",
    "SRECriterion",
    "UnitAlignedDistillationCriterion",
    "UGRIMDCriterion",
    "build_criterion",
    "criterion_list",
    "default_distillation_criterion",
]
