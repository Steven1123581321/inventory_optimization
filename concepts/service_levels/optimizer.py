import numpy as np
import pandas as pd
import os


class Optimizer:
    '''
    A class that performs dynamic programming to optimize service levels over n-items.
    '''

    items = None
    matrices = None
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
        items = []
        for index, row in self.data.iterrows():
            value_dictionary = {}
            m = int(min(self.demand_constraint, row.Demand))+1
            for alpha in range(m):
                service_level = min((row.Demand-alpha)/row.Demand, 1.)
                partial_expectation = (row.Quantity * (1. - service_level))/row.DDLT_Variation
                value = (4.85-(partial_expectation**1.3)*0.3924-(partial_expectation**0.135)*5.359)*row.DDLT_Variation*row.Inventory_Costs
                value_dictionary[alpha] = value
            items.append(value_dictionary)
        predecessor_values = []
        matrices = []
        for i in range(len(items)):
            if i > 0 and i < len(items)-1:
                matrix_values = {}
                minimum_alpha_values = []
                for alpha in items[i]:
                    elements = []
                    matrix_elements_dictionary = {}
                    for j in range(alpha+1):
                        if j <= alpha:
                            z = items[i][j]+predecessor[alpha-j]
                            if j == 0:
                                matrix_elements_dictionary[z] = alpha-j
                                x = z
                                p = z
                            elif p < z:
                                x = p
                            else:
                                matrix_elements_dictionary[z] = alpha-j
                                p = z
                                x = p
                    minimum_alpha_values.append(x)
                    matrix_values[alpha] = (matrix_elements_dictionary[x])
                matrices.append(matrix_values)
            elif i == len(items)-1:
                matrix_values = {items[i][key] + predecessor[min(self.data.iloc[i-1].Demand, self.demand_constraint)-key]: key for key in items[i]}
                matrices.append(matrix_values)
                self.matrices = matrices
                self.items = items
                return
            else:
                minimum_alpha_values = list(items[i].values())
                matrices.append({})
            predecessor = minimum_alpha_values

    def backward_pass(self):
        '''
        Does the calculations in the backward pass.
        The output is a dictionary with n-keys with the optimal service level per key (item).
        '''
        service_levels = {}
        for i in range(len(self.items)-1, -1, -1):
            if i == 0:
                service_levels[i] = max(0, self.data.iloc[i].Demand-take)/self.data.iloc[i].Demand
                self.service_levels = service_levels
                return
            if i == len(self.items)-1:
                leave = self.matrices[i][min(self.matrices[i].keys())]
                service_levels[i] = (self.data.iloc[i].Demand-leave)/self.data.iloc[i].Demand
                fill_rate_amount = service_levels[i]*self.data.iloc[i].Demand
            else:
                take = self.matrices[i][min(self.data.iloc[i].Demand, self.demand_constraint)-leave]
                service_levels[i] = (self.data.iloc[i].Demand-min(self.data.iloc[i].Demand, self.demand_constraint)+take+leave)/self.data.iloc[i].Demand
                fill_rate_amount = service_levels[i]*self.data.iloc[i].Demand
                leave += self.data.iloc[i].Demand-fill_rate_amount

    def group_service(self, data=None):
        '''
        Computes the overall service level to check if the algorithm has run correctly.
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
        self.data = data
        return value_own, value_teunter




