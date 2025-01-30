from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///canteen.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime, nullable=False)
    item_name = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    is_special_event = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<Sale {self.id}>'

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sales_entry', methods=['GET', 'POST'])
def sales_entry():
    if request.method == 'POST':
        try:
            date_time = datetime.strptime(request.form['date_time'], '%Y-%m-%dT%H:%M')
            item_name = request.form['item_name']
            quantity = int(request.form['quantity'])
            price = float(request.form['price'])
            is_special_event = 'is_special_event' in request.form

            if quantity <= 0 or price < 0:
                raise ValueError("Quantity must be positive and price must be non-negative.")

            new_sale = Sale(date_time=date_time, item_name=item_name, quantity=quantity, price=price, is_special_event=is_special_event)
            db.session.add(new_sale)
            db.session.commit()

            flash('Sale entry added successfully!', 'success')
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash('An error occurred while adding the sale entry.', 'error')
            app.logger.error(f"Error adding sale entry: {str(e)}")

        return redirect(url_for('sales_entry'))

    sales = Sale.query.order_by(Sale.date_time.desc()).all()
    return render_template('sales_entry.html', sales=sales)

@app.route('/edit_sale/<int:id>', methods=['GET', 'POST'])
def edit_sale(id):
    sale = Sale.query.get_or_404(id)
    if request.method == 'POST':
        try:
            sale.date_time = datetime.strptime(request.form['date_time'], '%Y-%m-%dT%H:%M')
            sale.item_name = request.form['item_name']
            sale.quantity = int(request.form['quantity'])
            sale.price = float(request.form['price'])
            sale.is_special_event = 'is_special_event' in request.form

            if sale.quantity <= 0 or sale.price < 0:
                raise ValueError("Quantity must be positive and price must be non-negative.")

            db.session.commit()
            flash('Sale entry updated successfully!', 'success')
            return redirect(url_for('sales_entry'))
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash('An error occurred while updating the sale entry.', 'error')
            app.logger.error(f"Error updating sale entry: {str(e)}")

    return render_template('edit_sale.html', sale=sale)

@app.route('/delete_sale/<int:id>', methods=['POST'])
def delete_sale(id):
    sale = Sale.query.get_or_404(id)
    try:
        db.session.delete(sale)
        db.session.commit()
        flash('Sale entry deleted successfully!', 'success')
    except Exception as e:
        flash('An error occurred while deleting the sale entry.', 'error')
        app.logger.error(f"Error deleting sale entry: {str(e)}")
    return redirect(url_for('sales_entry'))

@app.route('/sale_prediction', methods=['GET', 'POST'])
def sale_prediction():
    items = [item[0] for item in db.session.query(Sale.item_name).distinct()]
    if request.method == 'POST':
        future_date = request.form['future_date']
        future_time = request.form['future_time']
        item_name = request.form['item_name']
        is_special_event = 'is_special_event' in request.form
        future_datetime = datetime.strptime(f"{future_date} {future_time}", "%Y-%m-%d %H:%M")

        # Fetch all sales data for the selected item
        sales = Sale.query.filter_by(item_name=item_name).all()
        
        if not sales:
            flash('Not enough data for prediction. Please enter more sales data for this item.', 'error')
            return redirect(url_for('sale_prediction'))

        # Prepare data for prediction
        data = [(sale.date_time, sale.quantity, sale.is_special_event) for sale in sales]
        df = pd.DataFrame(data, columns=['date_time', 'quantity', 'is_special_event'])
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['hour'] = df['date_time'].dt.hour

        X = df[['day_of_week', 'month', 'hour', 'is_special_event']]
        y = df['quantity']

        # Create and train the model
        categorical_features = ['day_of_week', 'month', 'hour']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
            ])

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X, y)

        # Make prediction
        future_features = pd.DataFrame({
            'day_of_week': [future_datetime.weekday()],
            'month': [future_datetime.month],
            'hour': [future_datetime.hour],
            'is_special_event': [is_special_event]
        })

        predicted_quantity = model.predict(future_features)[0]

        return render_template('sale_prediction.html', prediction=round(predicted_quantity), items=items, selected_item=item_name)

    return render_template('sale_prediction.html', prediction=None, items=items)

@app.route('/sales_trend')
def sales_trend():
    sales = Sale.query.order_by(Sale.date_time).all()
    data = [{'date': sale.date_time.strftime('%Y-%m-%d'), 'quantity': sale.quantity, 'item': sale.item_name} for sale in sales]
    return render_template('sales_trend.html', sales_data=data)

@app.route('/api/sales_data')
def sales_data():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    item_name = request.args.get('item_name')

    query = Sale.query

    if start_date:
        query = query.filter(Sale.date_time >= datetime.strptime(start_date, '%Y-%m-%d'))
    if end_date:
        query = query.filter(Sale.date_time <= datetime.strptime(end_date, '%Y-%m-%d'))
    if item_name:
        query = query.filter(Sale.item_name == item_name)

    sales = query.order_by(Sale.date_time).all()
    data = [{'date': sale.date_time.strftime('%Y-%m-%d'), 'quantity': sale.quantity, 'item': sale.item_name} for sale in sales]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

