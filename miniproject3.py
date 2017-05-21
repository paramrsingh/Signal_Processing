# Param Singh
# Mini-Project 3

#Pairs trading using Kalman Filter

#Borrows heavily from https://www.quantopian.com/posts/ernie-chans-ewa-slash-ewc-pair-trade-with-kalman-filter, which borrowed from Ernie Chan https://www.amazon.com/Algorithmic-Trading-Winning-Strategies-Rationale/dp/1118460146

import numpy as np
import pytz


def initialize(context):
    
    #QQQ and DIA should move together, no cointegration check, building off Mini-Project1
    context.qqq = sid(19920)
    context.dia = sid(2174)
    
    #Left initialization of the matricies, noise and parameters intact
    context.delta = 0.0001
    context.Vw = context.delta / (1 - context.delta) * np.eye(2)
    context.Ve = 0.001
    context.beta = np.zeros(2)
    context.P = np.zeros((2, 2))
    context.R = None
    
    #trading position
    context.pos = None
    
    # Slippage and commission
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerTrade(cost=.75))
    
    schedule_function(mykalman, date_rule=date_rules.every_day(), time_rule=time_rules.market_close(hours=0, minutes=1))
    

def mykalman(context, data):
        
        x = np.asarray([data.current(context.qqq, 'price'), 1.0]).reshape((1, 2))
        y = data.current(context.dia, 'price')
    
        # update Kalman filter with latest price
        if context.R is not None:
            context.R = context.P + context.Vw
        else:
            context.R = np.zeros((2, 2))
        
        # Kalman calc, cribbed from the article
        yhat = x.dot(context.beta)
        Q = x.dot(context.R).dot(x.T) + context.Ve
        sqrt_Q = np.sqrt(Q)
        # e is the estimated spread as an output of the observation matrix and hyperparameters of the Kalman filter
        e = y - yhat
        #log.info(e)
        
        K = context.R.dot(x.T) / Q
        context.beta = context.beta + K.flatten() * e
        context.P = context.R - K * x.dot(context.R)
        
        #log.info(context.beta[0])
        #log.info(context.beta[1])
        
        # plot
        record(beta=context.beta[0], alpha=context.beta[1])
        if e < 5:
            record(spread=float(e), Q_upper=float(sqrt_Q), Q_lower=float(-sqrt_Q))
        
        # Update trading position
        if context.pos is not None:
            if context.pos == 'long' and e > -sqrt_Q:
                context.pos = None
            elif context.pos == 'short' and e < sqrt_Q:
                context.pos = None
        
        # Pairs-trading logic 
        if context.pos is None:
            if e < -sqrt_Q:
                log.info('opening long')
                order_percent(context.dia, 5)
                order_percent(context.qqq, -1 * context.beta[0])
                context.pos = 'long'
            elif e > sqrt_Q:
                log.info('opening short')
                order_percent(context.dia, -1)
                order_percent(context.qqq, 5 * context.beta[0])
                context.pos = 'short'