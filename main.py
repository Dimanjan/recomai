from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from collections import Counter
from typing import List

app = FastAPI(title="Product Recommendation API", description="API to recommend products based on customer similarity.")

# Load the pivot table and similarity matrix at startup
with open('models/pivot_df.pkl', 'rb') as f:
    pivot_df = pickle.load(f)

with open('models/similarity_df.pkl', 'rb') as f:
    similarity_df = pickle.load(f)

# Define the request body model
class RecommendationRequest(BaseModel):
    customer_name: str
    num_recommendations: int = 5

# Define the response model
class RecommendationResponse(BaseModel):
    customer_name: str
    recommended_products: List[str]

def recommend_products(customer_name: str, num_recommendations: int = 5) -> List[str]:
    if customer_name not in similarity_df.index:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_name}' not found.")

    similar_customers = similarity_df[customer_name].sort_values(ascending=False).index[1:]  # Exclude self
    product_counter = Counter()

    for customer in similar_customers:
        purchased_products = pivot_df.loc[customer][pivot_df.loc[customer] > 0].index.tolist()
        product_counter.update(purchased_products)

        if len(product_counter) >= num_recommendations * 2:
            # Fetch more to ensure uniqueness after exclusion
            break

    # Exclude products already purchased by the target customer
    target_purchased = pivot_df.loc[customer_name][pivot_df.loc[customer_name] > 0].index.tolist()
    for product in target_purchased:
        if product in product_counter:
            del product_counter[product]

    recommended = [product for product, _ in product_counter.most_common(num_recommendations)]
    return recommended

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    recommended = recommend_products(request.customer_name, request.num_recommendations)
    return RecommendationResponse(
        customer_name=request.customer_name,
        recommended_products=recommended
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Product Recommendation API. Use the /recommend endpoint to get product recommendations."}

@app.get("/customers", response_model=List[str])
def list_customers():
    return pivot_df.index.tolist()

@app.get("/status")
def get_status():
    return {"status": "API is running."}
