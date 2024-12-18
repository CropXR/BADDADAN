import logging

import click
from pathlib import Path
from experiment_scripts import pypesto_from_sbml

@click.command()
@click.argument('treatment_path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('treatment_name', type=str)
@click.argument('expr_mat_pkl_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('sbml_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(treatment_path, treatment_name, expr_mat_pkl_path, sbml_path):
    """
    Run PyPESTO from SBML with treatment data.

    TREATMENT_PATH: Path to the treatment directory.
    TREATMENT_NAME: Name of the treatment.
    SBML_PATH: Path to the SBML file.
    """
    (treatment_path / "log.log").unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(treatment_path / "log.log"),
                                  logging.StreamHandler()])

    pypesto_from_sbml(
        treatment_path,
        treatment_name,
        expr_mat_pkl_path,
        sbml_path,
        do_ml_flow_logging=False
    )


if __name__ == '__main__':
    main()
