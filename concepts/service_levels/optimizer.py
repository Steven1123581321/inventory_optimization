import numpy as np
import pandas as pd
import os


class Optimizer:
    '''
    A class that performs dynamic programming to optimize service levels over n-items.
    '''
    core_values = None
    values = None
    pieces = None

    def __init__(self, constraint=None, data=None):
        if constraint is None:
            raise Exception(
                'No constraint is provided. Please provide a feasible constraint.'
            )
        if not 0. <= constraint < 1. or not isinstance(constraint, float):
            raise Exception(
                f'{constraint} is not a feasible constraint. Please provide a float between zero and 1.'
            )
        if data is None:
            raise Exception(
                f'No data is provided, thus no data can be loaded.'
            )
        # Save models and model_parameters.
        self.constraint = constraint
        self.data = data


    @classmethod
    def load(cls, file_name, directory):
        '''
        Load data from an excel file.
        '''
        # Check filename
        _, extension = os.path.splitext(file_name)
        if extension != '.xlsx':
            raise Exception(
                'File does not have the extension ".xlsx", but {}.'.format(extension) + ' Please provide an excel-file.'
            )
        # Load data
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'rb') as input_file:
            data = pd.read_excel(input_file)
        if len(data.columns) != 5:
            raise Exception(
                f'The data appears to not have the right amount of variables.'+\
                ' Please check your file on columns that are outside {list(data.columns)}'
            )
        if data.isnull().values.any():
            raise Exception(
                f'The data appears to have missing values in it.'+\
                ' Please check your data on missing values.'
            )
        return data

    def forward_pass(self):
        '''
        Do the calculations in the forward pass.
        The output is a list with n-dictionaries, 
        containing the safety stock value and the corresponding service level.
        '''
        aggregated_demand = np.sum(self.data.Demand)
        demand_constraint = int((1-self.constraint)*aggregated_demand)
        values = []
        pieces = []
        for index, row in self.data.iterrows():
            dict_1 = {}
            dict_2 = {}
            for piece in range(int(np.round(min(demand_constraint, row.Demand),0))+1):
                service_level = min((row.Demand-piece)/row.Demand, 1.)
                partial_expectation = (row.Quantity * (1. - service_level))/row.DDLT_Variation
                value = np.round((4.85-(partial_expectation**1.3)*0.3924-(partial_expectation**0.135)*5.359)*row.DDLT_Variation*row.Inventory_Costs, 2)
                dict_1[value] = service_level
                dict_2[piece] = value
            values.append(dict_1)
            pieces.append(dict_2)
        focus_values = []
        core_values = []
        for i in range(len(pieces)):
            if i == 0:
                focus_values.append(list(pieces[i].values()))
            elif i == len(pieces)-1:
                focus_values.append(list(pieces[i].values()))
            else:
                fractions = []
                for key in pieces[i]:
                    quarks = []
                    for j in range(key+1):
                        if j <= key:
                            z = pieces[i][j]+predecessor[key-j]
                            quarks.append(z)
                        x = min(quarks)
                    fractions.append(x)
                focus_values.append(fractions)
            predecessor = {}
            for k in range(len(focus_values[i])):
                predecessor[k] = focus_values[i][k]
            core_values.append(predecessor)
        self.core_values = core_values
        self.values = values
        self.pieces = pieces









