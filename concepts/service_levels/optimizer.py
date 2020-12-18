import numpy as np
import pandas as pd
import os


class Optimizer:
    '''
    A class that performs dynamic programming to optimize service levels over n-items.
    '''

    matrix = None
    B = None
    service_levels = None
    N = None

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
        '''
        aggregated_demand = np.sum(self.data.Demand)
        demand_constraint = int((1-self.constraint)*aggregated_demand)
        self.B = demand_constraint
        self.N = len(self.data)
        value_function = lambda x, Q, D, s, h: (4.85-((Q*(1-min(x/D, 1.))/s)**1.3)*0.3924-((Q*(1-min(x/D, 1.))/s)**0.135)*5.359)*s*h
        matrix = []
        for i in range(self.N):
            if i > 0 and i < self.N-1:
                vector_1 = {}
                vector_2 = {}
                for alpha in range(self.B+1):
                    Q = self.data.iloc[i].Quantity
                    D = self.data.iloc[i].Demand
                    s = self.data.iloc[i].DDLT_Variation
                    h = self.data.iloc[i].Inventory_Costs
                    for beta in range(alpha, -1, -1):
                        scalar = {alpha:value_function(max(D-alpha+beta, 0), Q, D, s, h)+predecessor[beta]}
                        if beta == alpha:
                            scalar_value = scalar[alpha]
                            beta_value = {alpha:beta}
                        elif previous_scalar < scalar[alpha]:
                            beta_value = {alpha:beta+1}
                            scalar = {alpha:previous_scalar}
                            break
                        else:
                            beta_value = {alpha:beta}
                            pass
                        previous_scalar = scalar[alpha]
                    vector_1.update(scalar)
                    vector_2.update(beta_value)
            elif i == self.N-1:
                vector_2 = []
                for alpha in range(min(self.data.iloc[i].Demand+1, self.B+1)):
                    Q = self.data.iloc[i].Quantity
                    D = self.data.iloc[i].Demand
                    s = self.data.iloc[i].DDLT_Variation
                    h = self.data.iloc[i].Inventory_Costs
                    scalar = {alpha:value_function(D-alpha, Q, D, s, h)+predecessor[self.B-alpha]}
                    if alpha == 0:
                        previous_scalar = scalar[alpha]
                        alpha_value = alpha
                    elif alpha == min(self.data.iloc[i].Demand, self.B):
                        alpha_value = alpha
                        break
                    elif previous_scalar < scalar[alpha]:
                        alpha_value = alpha-1
                        scalar = {alpha:previous_scalar}
                        break
                    else:
                        previous_scalar = scalar[alpha]
                vector_2.append(alpha_value)
            else:
                vector_1 = {}
                vector_2 = {}
                for alpha in range(self.B+1):
                    Q = self.data.iloc[i].Quantity
                    D = self.data.iloc[i].Demand
                    s = self.data.iloc[i].DDLT_Variation
                    h = self.data.iloc[i].Inventory_Costs
                    scalar = {alpha:value_function(max(D-alpha, 0), Q, D, s, h)}
                    beta_value = {alpha:alpha}
                    vector_1.update(scalar)
                    vector_2.update(beta_value)
            matrix.append(vector_2)
            predecessor = vector_1
        self.matrix = matrix

    def backward_pass(self):
        '''
        Does the calculations in the backward pass.
        '''
        self.service_levels = {}
        for i in range(self.N-1, -1, -1):
            if i > 0 and i < self.N-1:
                backorders_spending_on_this_item = backorders_to_spend-self.matrix[i][backorders_to_spend]
                self.service_levels[i] = (self.data.iloc[i].Demand-backorders_spending_on_this_item)/self.data.iloc[i].Demand
                backorders_to_spend = self.matrix[i][backorders_to_spend]
            elif i == self.N-1:
                backorders_spending_on_this_item = self.matrix[i][0]
                self.service_levels[i] = (self.data.iloc[i].Demand-backorders_spending_on_this_item)/self.data.iloc[i].Demand
                backorders_to_spend = self.B - backorders_spending_on_this_item 
            else:
                self.service_levels[i] = (self.data.iloc[i].Demand-backorders_to_spend)/self.data.iloc[i].Demand

    def group_service(self, data=None):
        '''
        Computes the overall service level to check if the algorithm has run correctly.
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