from typer import Typer
from neuralab.trading import dataframe, dataset


cli = Typer(name="neuralab")
cli.add_typer(dataframe.cli, name="dataframe")
cli.add_typer(dataset.cli, name="dataset")
cli()