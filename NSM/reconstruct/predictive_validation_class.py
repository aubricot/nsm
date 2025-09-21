import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


class Regress:
    """
    Class for performing regression on a set of latent variables

    list_factors: list of strings, each string is a factor name
    list_paths: list of strings, each string is a path to a file
    """

    def __init__(self, list_factors, list_paths) -> None:
        """
        list_factors: list of strings, each string is a factor name
        list_paths: list of strings, each string is a path to a file
        """

        self.list_factors = list_factors
        self.list_paths = list_paths

        self.list_latents = []
        self.latents = None

        self.df = self.get_all_factors()

    def add_latent(self, latent):
        """
        latent: list of floats, each float is a latent variable

        Add a latent variable to the list of latents
        """
        self.list_latents.append(latent)

    def calc_r2(self):
        """
        Return the R^2 for a regression model between
        all the latent variables and each factor

        Return a list of R^2 values
        """
        dict_results = {}

        self.latents = np.array(self.list_latents)
        for factor in self.list_factors:
            r2 = self.calc_r2_single_factor(factor)
            dict_results[f"val_prediction_{factor}"] = r2
        return dict_results

    def calc_r2_single_factor(self, factor):
        """
        factor: string, the name of a factor

        Return the R^2 for a regression model between
        all the latent variables and the given factor

        Return a float
        """
        # Get the factors
        dv = self.df[factor]

        # Create a linear regression object
        reg = LinearRegression()
        # Fit the regression model
        reg.fit(self.latents, dv)
        # Get the R^2 score
        r2 = reg.score(self.latents, dv)

        return r2

    def get_factors_from_filename(self, filename):
        """
        filename: string, the filename of a file

        Extract the variables from the filename
        Return the variables as a dictionary
        """

        variables = {}
        for varname in self.list_factors:
            variables[varname] = float(filename.split(f"{varname}_")[1].split("-")[0])

        return variables

    def get_all_factors(self):
        """
        Return a dataframe of all the factors
        """
        # Get the factors for all files
        all_factors = []
        for filename in self.list_paths:
            all_factors.append(self.get_factors_from_filename(filename))
        # Return the factors as a list of dictionaries

        df = pd.DataFrame(all_factors)
        return df
