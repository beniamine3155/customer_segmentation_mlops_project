from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.pipline.prediction_pipeline import (
        CustomerSegmentationData,
        CustomerSegmentationClassifier,
        predict_customer_segment
    )
    from src.logger import logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in demo mode without actual prediction pipeline")
    logging = None

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation Predictor",
    description="Predict customer segments based on shopping behavior using Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files (optional - CSS is embedded in template)
import os
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age (18-100 years)")
    income: float = Field(..., ge=0, description="Annual income in dollars")
    total_spend: float = Field(..., ge=0, description="Total annual spending in dollars")
    recency: int = Field(..., ge=0, le=365, description="Days since last purchase")
    num_web_purchases: int = Field(..., ge=0, description="Number of web purchases")
    num_store_purchases: int = Field(..., ge=0, description="Number of store purchases")
    num_web_visits_month: int = Field(..., ge=0, description="Number of website visits per month")

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000,
                "total_spend": 1200,
                "recency": 30,
                "num_web_purchases": 5,
                "num_store_purchases": 3,
                "num_web_visits_month": 4
            }
        }

class PredictionResponse(BaseModel):
    cluster_id: int
    segment_name: str
    segment_description: str
    customer_data: dict
    confidence: str = "High"

class ErrorResponse(BaseModel):
    error: str
    message: str

# Segment mapping for demo mode
SEGMENT_MAPPING = {
    0: {"name": "Budget_Conscious", "description": "Price-sensitive customers who focus on value and deals"},
    1: {"name": "High_Value", "description": "High income customers with significant spending power"},
    2: {"name": "Regular_Customers", "description": "Consistent customers with moderate spending patterns"},
    3: {"name": "Premium_Shoppers", "description": "Quality-focused customers who prefer premium products"},
    4: {"name": "Occasional_Buyers", "description": "Customers who make infrequent but meaningful purchases"},
    5: {"name": "Loyal_Customers", "description": "Highly engaged customers with frequent interactions"}
}

def simulate_prediction(customer_data: CustomerData) -> PredictionResponse:
    """
    Simulate prediction when actual model is not available
    """
    # Simple rule-based prediction for demo
    if customer_data.income > 70000 and customer_data.total_spend > 2000:
        cluster_id = 1  # High Value
    elif customer_data.total_spend > 1500 and customer_data.num_web_purchases < 3:
        cluster_id = 3  # Premium Shoppers
    elif customer_data.recency > 60 or (customer_data.num_web_purchases + customer_data.num_store_purchases) < 3:
        cluster_id = 4  # Occasional Buyers
    elif (customer_data.num_web_purchases + customer_data.num_store_purchases) > 10 and customer_data.num_web_visits_month > 5:
        cluster_id = 5  # Loyal Customers
    elif customer_data.total_spend < 800:
        cluster_id = 0  # Budget Conscious
    else:
        cluster_id = 2  # Regular Customers

    segment = SEGMENT_MAPPING[cluster_id]
    
    return PredictionResponse(
        cluster_id=cluster_id,
        segment_name=segment["name"],
        segment_description=segment["description"],
        customer_data=customer_data.dict(),
        confidence="High"
    )

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serve the main page
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict_segment(customer_data: CustomerData):
    """
    Predict customer segment based on input data
    """
    try:
        if logging:
            logging.info(f"Received prediction request: {customer_data}")
        
        # Validate spending vs income
        if customer_data.total_spend > customer_data.income:
            raise HTTPException(
                status_code=400,
                detail="Total spending cannot exceed annual income"
            )
        
        try:
            # Try to use actual prediction pipeline
            result = predict_customer_segment(
                age=customer_data.age,
                income=customer_data.income,
                total_spend=customer_data.total_spend,
                recency=customer_data.recency,
                num_web_purchases=customer_data.num_web_purchases,
                num_store_purchases=customer_data.num_store_purchases,
                num_web_visits_month=customer_data.num_web_visits_month
            )
            
            return PredictionResponse(
                cluster_id=result['predicted_cluster'],
                segment_name=result['segment_details']['Segment_Name'],
                segment_description=SEGMENT_MAPPING.get(result['predicted_cluster'], {}).get('description', 'Customer segment'),
                customer_data=result['customer_data'],
                confidence=result['prediction_confidence']
            )
            
        except Exception as model_error:
            if logging:
                logging.warning(f"Model prediction failed: {model_error}. Using simulation.")
            
            # Fall back to simulation
            return simulate_prediction(customer_data)
            
    except HTTPException:
        raise
    except Exception as e:
        if logging:
            logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-form")
async def predict_form(request: Request):
    """
    Handle form submission from HTML
    """
    try:
        form_data = await request.form()
        
        # Convert form data to CustomerData
        customer_data = CustomerData(
            age=int(form_data.get("age")),
            income=float(form_data.get("income")),
            total_spend=float(form_data.get("total_spend")),
            recency=int(form_data.get("recency")),
            num_web_purchases=int(form_data.get("num_web_purchases")),
            num_store_purchases=int(form_data.get("num_store_purchases")),
            num_web_visits_month=int(form_data.get("num_web_visits_month"))
        )
        
        # Get prediction
        prediction = await predict_segment(customer_data)
        
        # Return template with results
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "result": {
                    "cluster_id": prediction.cluster_id,
                    "segment_name": prediction.segment_name,
                    "description": prediction.segment_description,
                    "customer_data": prediction.customer_data
                }
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error processing prediction: {str(e)}"
            }
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "Customer Segmentation API is running"}

@app.get("/segments")
async def get_segments():
    """
    Get information about all customer segments
    """
    return {
        "segments": SEGMENT_MAPPING,
        "total_segments": len(SEGMENT_MAPPING)
    }

@app.post("/predict-batch")
async def predict_batch(customers: list[CustomerData]):
    """
    Predict segments for multiple customers
    """
    try:
        if len(customers) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 customers allowed per batch"
            )
        
        results = []
        for customer in customers:
            try:
                prediction = await predict_segment(customer)
                results.append(prediction)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "customer_data": customer.dict()
                })
        
        return {
            "predictions": results,
            "total_processed": len(customers),
            "successful_predictions": len([r for r in results if not hasattr(r, 'error')])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "error": "Page not found"},
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "error": "Internal server error"},
        status_code=500
    )

# Main function
if __name__ == "__main__":
    # Check if model files exist
    model_available = os.path.exists("src/pipline/prediction_pipeline.py")
    
    if model_available:
        print("✅ Model pipeline detected - Running with full prediction capability")
    else:
        print("⚠️  Model pipeline not found - Running in demo mode")
    
    print("\n🚀 Starting Customer Segmentation API...")
    print("📊 Available endpoints:")
    print("   • GET  /          - Web interface")
    print("   • POST /predict   - API prediction")
    print("   • POST /predict-form - Form submission")
    print("   • GET  /health    - Health check")
    print("   • GET  /segments  - Segment information")
    print("   • GET  /docs      - API documentation")
    print("   • GET  /redoc     - Alternative API docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        reload_dirs=[".", "src", "templates", "static"]
    )