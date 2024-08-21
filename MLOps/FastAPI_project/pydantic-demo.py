from pydantic import BaseModel, PositiveInt
from typing import Literal, List, Dict, Tuple

class Animal(BaseModel):
    name: str
    color: Literal['Golden', 'White']
    age: int
    owner: Dict[str, PositiveInt] # name age
    breed: str = "German"

dog1 = {
    'name': 'Husky',
    'color': 'Golden',
    'age': 12,
    'owner': {'Harsh': 25, 'Yash': 22},
    'breed': 'American'
}

husky = Animal(**dog1)

print(husky.age, husky.owner)
print(husky.model_dump())


