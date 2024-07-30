from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

app = Flask(__name__)

model_file = open('stokBarang.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', purchItem=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the purchItem based on user inputs
    and render the result to the html page
    '''
    dscription, uom1, uom2, uom3, lstSalDate, lstPurDate, regDate, objType, uom4 = [x for x in request.form.values()]

    data = []
    x_trans = OrdinalEncoder()
    data.append(dscription)
    data.append(str(uom1))
    data.append(str(uom2))
    data.append(str(uom3))
    data.append(str(lstSalDate))
    data.append(str(lstPurDate))
    data.append(str(regDate))
    data.append(str(objType))
    data.append(str(uom4))
    
    print(data)
          
    # #mengkodekan semua value menjadi ordinal
  

    x_trans = OrdinalEncoder()
    X = x_trans.fit_transform(data)

    print(X)      
    X = np.reshape(1, -1)   

    print(X)

    prediction = model.predict([X])

    output = round(prediction[0], 1)

    return render_template('index.html', purchItem = output,  
                       dscription=dscription, uom1 = uom1, uom2 = uom2, uom3 = uom3,
                       lstSalDate = lstSalDate, lstPurDate = lstPurDate, 
                       regDate = regDate, objType = objType, uom4 = uom4)


if __name__ == '__main__':
    app.run(debug=True)