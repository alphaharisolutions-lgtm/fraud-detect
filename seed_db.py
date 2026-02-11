from app import app, db, Transaction
import random

def seed_data():
    with app.app_context():
        # Clear existing data
        db.drop_all() 
        db.create_all()
        
        merchants = ["Amazon", "Flipkart", "Uber", "Swiggy", "Zomato", "Reliance", "Starbucks"]
        locations = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Others"]
        
        for i in range(50):
            amount = random.uniform(100, 20000)
            is_fraud = amount > 15000 or (amount > 5000 and random.random() > 0.8)
            
            t = Transaction(
                amount=round(amount, 2),
                utr_number=f"UTR-{random.randint(1000000000, 9999999999)}",
                location=random.choice(locations),
                time=f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
                frequency=random.randint(1, 10),
                is_fraud=is_fraud,
                prediction_score=random.uniform(0.7, 0.99) if is_fraud else random.uniform(0.01, 0.3)
            )
            db.session.add(t)
        
        db.session.commit()
        print("Database seeded with 50 transactions.")

if __name__ == "__main__":
    seed_data()
