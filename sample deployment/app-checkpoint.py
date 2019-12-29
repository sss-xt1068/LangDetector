{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "import pickle\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = pickle.load(open('model.pkl', 'rb'))\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "\n",
    "    int_features = [int(x) for x in request.form.values()]\n",
    "    final_features = [np.array(int_features)]\n",
    "    prediction = model.predict(final_features)\n",
    "\n",
    "    output = round(prediction[0], 2)\n",
    "\n",
    "    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))\n",
    "\n",
    "@app.route('/results',methods=['POST'])\n",
    "def results():\n",
    "\n",
    "    data = request.get_json(force=True)\n",
    "    prediction = model.predict([np.array(list(data.values()))])\n",
    "\n",
    "    output = prediction[0]\n",
    "    return jsonify(output)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: flask in c:\\users\\hp\\anaconda3\\lib\\site-packages (1.1.1)\n",
      "Requirement already satisfied: itsdangerous>=0.24 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from flask) (1.1.0)\n",
      "Requirement already satisfied: Werkzeug>=0.15 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from flask) (0.16.0)\n",
      "Requirement already satisfied: click>=5.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from flask) (7.0)\n",
      "Requirement already satisfied: Jinja2>=2.10.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from flask) (2.10.3)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from Jinja2>=2.10.1->flask) (1.1.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install flask"
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
