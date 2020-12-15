import numpy as np
import pandas as pd
import os


class Optimizer:
    '''
    A class that performs dynamic programming to optimize service levels over n-items.
    '''

    pieces = None
    matrix_values = None
    demand_constraint = None
    service_levels = None

    def __init__(self, constraint=None, data=None):
        '''
        Initialize and check created instance
        '''
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
        Load and check data from an excel file.
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
        Does the calculations in the forward pass.
        The main output is a list with 2 x n-dictionaries:
        1.
        Containing the safety stock value and the corresponding service level.
        2.
        Containing the safety stock value and a tuple,
        indicating how many pieces we take to the next period
        and how many we have left in this period.
        '''
        aggregated_demand = np.sum(self.data.Demand)
        demand_constraint = int((1-self.constraint)*aggregated_demand)
        self.demand_constraint = demand_constraint
        pieces = []
        for index, row in self.data.iterrows():
            dict_1 = {}
            for piece in range(int(np.round(min(self.demand_constraint, row.Demand),0))+1):
                service_level = min((row.Demand-piece)/row.Demand, 1.)
                partial_expectation = (row.Quantity * (1. - service_level))/row.DDLT_Variation
                value = np.round((4.85-(partial_expectation**1.3)*0.3924-(partial_expectation**0.135)*5.359)*row.DDLT_Variation*row.Inventory_Costs, 2)
                dict_1[piece] = value
            pieces.append(dict_1)
        focus_values = []
        matrix = []
        for i in range(len(pieces)):
            if i == 0:
                focus_values.append(list(pieces[i].values()))
                matrix_values = {}
                for key in pieces[i]:
                    matrix_values[pieces[i][key]] = (0, key)
                matrix.append(matrix_values)
            elif i == len(pieces)-1:
                matrix_values = {}
                for key in pieces[i]:
                    matrix_values[pieces[i][key] + predecessor[min(self.data.iloc[i-1].Demand, self.demand_constraint)-key]] = key
                matrix.append(matrix_values)
                self.matrix_values = matrix
                self.pieces = pieces
                return
            else:
                matrix_values = {}
                fractions = []
                for key in pieces[i]:
                    quarks = []
                    dict_2 = {}
                    for j in range(key+1):
                        if j <= key:
                            z = pieces[i][j]+predecessor[key-j]
                            quarks.append(z)
                            dict_2[z] = key-j
                        x = min(quarks)
                    fractions.append(x)
                    matrix_values[key] = (dict_2[x])
                focus_values.append(fractions)
                matrix.append(matrix_values)
            predecessor = {}
            for k in range(len(focus_values[i])):
                predecessor[k] = focus_values[i][k]

    def backward_pass(self):
        '''
        Does the calculations in the backward pass.
        The output is a dictionary with n-keys with the optimal service level per key (item).
        '''
        service_levels = {}
        for i in range(len(self.pieces)-1, -1, -1):
            if i == 0:
                service_levels[i] = max(0, self.data.iloc[i].Demand - (self.demand_constraint-take_with_me))/self.data.iloc[i].Demand
                self.service_levels = service_levels
                return
            if i == len(self.pieces)-1:
                x = self.matrix_values[i][min(self.matrix_values[i].keys())]
                service_levels[i] = (self.data.iloc[i].Demand-x)/self.data.iloc[i].Demand
                demand = service_levels[i]*self.data.iloc[i].Demand
                take_with_me = self.data.iloc[i].Demand-demand
            else:
                y = self.matrix_values[i][min(self.data.iloc[i].Demand, self.demand_constraint)-take_with_me]
                service_levels[i] = (self.data.iloc[i].Demand-min(self.data.iloc[i].Demand, self.demand_constraint)+y+take_with_me)/self.data.iloc[i].Demand
                demand = service_levels[i]*self.data.iloc[i].Demand
                take_with_me += self.data.iloc[i].Demand-demand

    def group_service(self, data=None):
        '''
        Computers the overall service level to check if the algorithm has run correctly.
        The output is a return of the group service level.
        '''
        if data is None:
            raise Exception(
                f'No data is provided, thus no data can be loaded.'
            )
        group_service = sum(data.Service_Level*data.Demand)/sum(data.Demand)
        return group_service

    def teunter(self, data=None, holding_cost_percentage = 0.25, to_confirm=False):
        '''
        Executes the algorithm of Teunter et al.
        The output is the self.data attribute with the service levels from the algorithm.
        '''
        if to_confirm:
            response = input('We assume a holding cost percentage of 25%. Do you agree? [y/n] ')
            if not response == 'y':
                return
        if data is None:
            raise Exception(
                f'No data is provided, thus no data can be loaded.'
            )
        data['Price'] = data.apply(lambda row: row.Inventory_Costs/holding_cost_percentage, axis=1)
        APCR = sum(data.apply(lambda row: row.Price*(row.Demand/sum(data.Demand)), axis=1))
        data['Service_Level_Teunter'] = data.apply(lambda row: 1-(1-self.constraint)*(row.Price/APCR), axis=1)
        self.data = data
        return

    def calculate_costs(self, data=None, number_of_algorithms=2):
        '''
        Executes the cost calculation for both service levels.
        The output is the cost calculation for our algorithm and Teunter's.
        '''
        data = self.data
        if data is None:
            raise Exception(
                f'First perform the algorithms first.'
            )
        pe_array = np.array(data.apply(lambda row: (row.Quantity * (1. - row.Service_Level))/row.DDLT_Variation, axis=1))
        value_own = sum((4.85-(pe_array**1.3)*0.3924-(pe_array**0.135)*5.359)*data.DDLT_Variation*data.Inventory_Costs)
        pe_array = np.array(data.apply(lambda row: (row.Quantity * (1. - row.Service_Level_Teunter))/row.DDLT_Variation, axis=1))
        value_teunter = sum((4.85-(pe_array**1.3)*0.3924-(pe_array**0.135)*5.359)*data.DDLT_Variation*data.Inventory_Costs)
        self.data=data
        return value_own, value_teunter




