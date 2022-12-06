
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

app = FastAPI()

model = pickle.load(open("emp.pkl", "rb"))
enc = pickle.load(open('encoder.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))


class Data(BaseModel):
    age: int
    education: int
    environmentsatisfaction: int
    jobinvolvement: int
    joblevel: int
    jobsatisfaction: int
    Annualincome: int
    relationshipsatisfaction: int
    totalworkingyearsexperience: int
    TrainingTime: int
    WorkLifeBalance: int
    BehaviourialCompetence: int
    OntimeDelivery: int
    TicketSolvingManagements: int
    Projectevlaution: int
    Psychosocialindicators: int
    PercentSalaryHike: int
    Netconnectivity: str
    gender: str
    maritalstatus: str
    department: str
    jobrole: str
    Workingfromhomeoroffice: str
    overtime: str
    attendance: str
    effectedwithcorona: str


@app.post("/")
def emp_per(data: Data):

    data_dict = data.dict()
    to_predict = [data_dict[feature] for feature in features]

    encoded_features = list(enc.transform(np.array(to_predict[-9:]).reshape(1, -1))[0])
    to_predict = np.array(to_predict[:-9] + encoded_features)

    prediction = model.predict(to_predict.reshape(1, -1))

    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
