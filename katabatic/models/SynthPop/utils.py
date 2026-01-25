"""
Utility functions for SynthPop integration in Katabatic.
"""

from pathlib import Path


def write_r_script(
    r_script_path: Path,
    input_csv: Path,
    output_csv: Path,
    seed: int
):
    

    r_code = f"""
    suppressMessages(library(synthpop))

    set.seed({seed})

    data <- read.csv("{input_csv.as_posix()}")

    syn_data <- syn(
        data,
        method = "cart",
        seed = {seed}
    )

    write.csv(
        syn_data$syn,
        "{output_csv.as_posix()}",
        row.names = FALSE
    )
    """

    r_script_path.write_text(r_code.strip())
