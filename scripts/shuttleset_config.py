"""ShuttleSet configuration: shot type classification and player metadata.

Chinese-to-English translation from the ShuttleSet README.
Classification into attack/neutral/safe is a modelling decision — adjust as needed.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Chinese → English → rally-style category
# --------------------------------------------------------------------------- #

CHINESE_TO_ENGLISH: dict[str, str] = {
    "發短球": "short service",
    "發長球": "long service",
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過度切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "未知球種": "unknown",
}

# Rally-style classification (serves handled separately)
SHOT_CLASSIFICATION: dict[str, str] = {
    # ATTACK — aggressive shots intended to win rally or force weak return
    "smash": "attack",
    "wrist smash": "attack",
    "rush": "attack",
    "push": "attack",
    "drop": "attack",
    # NEUTRAL — shots that maintain rally position
    "drive": "neutral",
    "driven flight": "neutral",
    "back-court drive": "neutral",
    "clear": "neutral",
    "cross-court net shot": "neutral",
    # SAFE — defensive or tempo-controlling shots
    "lob": "safe",
    "defensive return lob": "safe",
    "defensive return drive": "safe",
    "net shot": "safe",
    "return net": "safe",
    "passive drop": "safe",
}

SERVE_TYPES = {"short service", "long service"}

# --------------------------------------------------------------------------- #
# Player metadata (manually verified from BWF profiles)
# --------------------------------------------------------------------------- #

PLAYER_COUNTRY: dict[str, str] = {
    "Kento MOMOTA": "Japan",
    "CHOU Tien Chen": "Chinese Taipei",
    "CHEN Long": "China",
    "NG Ka Long Angus": "Hong Kong",
    "SHI Yuqi": "China",
    "Viktor AXELSEN": "Denmark",
    "Anders ANTONSEN": "Denmark",
    "Jonatan CHRISTIE": "Indonesia",
    "Anthony Sinisuka GINTING": "Indonesia",
    "LEE Zii Jia": "Malaysia",
    "Rasmus GEMKE": "Denmark",
    "KIDAMBI Srikanth": "India",
    "An Se Young": "South Korea",
    "Ratchanok INTANON": "Thailand",
    "Carolina MARIN": "Spain",
    "Pornpawee CHOCHUWONG": "Thailand",
    "Mia BLICHFELDT": "Denmark",
    "Busanan ONGBAMRUNGPHAN": "Thailand",
    "Supanida KATETHONG": "Thailand",
    "Neslihan YIGIT": "Turkey",
    "PUSARLA V. Sindhu": "India",
    "Evgeniya KOSETSKAYA": "Russia",
    "Michelle LI": "Canada",
    "LEE Cheuk Yiu": "Hong Kong",
    "Hans-Kristian Solberg VITTINGHUS": "Denmark",
    "Sameer VERMA": "India",
    "LIEW Daren": "Malaysia",
}

PLAYER_HANDEDNESS: dict[str, str] = {
    "Kento MOMOTA": "R",
    "CHOU Tien Chen": "R",
    "CHEN Long": "R",
    "NG Ka Long Angus": "R",
    "SHI Yuqi": "R",
    "Viktor AXELSEN": "R",
    "Anders ANTONSEN": "R",
    "Jonatan CHRISTIE": "R",
    "Anthony Sinisuka GINTING": "R",
    "LEE Zii Jia": "R",
    "Rasmus GEMKE": "R",
    "KIDAMBI Srikanth": "R",
    "An Se Young": "R",
    "Ratchanok INTANON": "R",
    "Carolina MARIN": "L",
    "Pornpawee CHOCHUWONG": "R",
    "Mia BLICHFELDT": "R",
    "Busanan ONGBAMRUNGPHAN": "R",
    "Supanida KATETHONG": "R",
    "Neslihan YIGIT": "R",
    "PUSARLA V. Sindhu": "R",
    "Evgeniya KOSETSKAYA": "R",
    "Michelle LI": "R",
    "LEE Cheuk Yiu": "R",
    "Hans-Kristian Solberg VITTINGHUS": "R",
    "Sameer VERMA": "R",
    "LIEW Daren": "R",
}
