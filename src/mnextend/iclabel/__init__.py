# © MNEXTEND developers
#
# License: BSD (3-clause)

from mnextend.iclabel.iclabel import run_iclabel

IC_LABELS = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]

__all__ = ["IC_LABELS", "run_iclabel"]
