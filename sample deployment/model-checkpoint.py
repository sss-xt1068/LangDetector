{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[143.3072588]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "dataset = pd.read_csv('C:/Users/HP/Desktop/sample deployment/sales.csv')\n",
    "\n",
    "dataset['rate'].fillna(0, inplace=True)\n",
    "\n",
    "dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)\n",
    "\n",
    "X = dataset.iloc[:, :3]\n",
    "\n",
    "def convert_to_int(word):\n",
    "    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,\n",
    "                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}\n",
    "    return word_dict[word]\n",
    "\n",
    "X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))\n",
    "\n",
    "y = dataset.iloc[:, -1]\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "regressor = LinearRegression()\n",
    "\n",
    "regressor.fit(X, y)\n",
    "\n",
    "pickle.dump(regressor, open('model.pkl','wb'))\n",
    "\n",
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[4, 300, 500]]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
