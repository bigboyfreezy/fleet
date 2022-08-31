
from flask import *
import joblib
import pandas

from flask import request
app.secret_key = 'A+4#s_T%P8g0@o?6'

import pymysql
app = Flask(__name__)





def send():
    conn = pymysql.connect(host='localhost',user = 'root',password='',database='sierra')
    mwanzo = 'Nairobi'
    mwisho = 'Kisii'
    date = '22/2/20220'
    time = '20:00'
    amount = 500

    
    cursor = conn.cursor()
    sql = "insert into bookings(mwanzo,mwisho,date,time,amount)values(%s,%s,%s,%s,%s)"
    try:
        cursor.execute(sql,(mwanzo,mwisho,date,time,amount))
        conn.commit()
        print('succesfull')
        
    except:
        print('Fail')
    

send()







@app.route ('/book',methods=['POST','GET'])
def book():
    
    if request.method == 'POST':
        mwanzo = request.form['mwanzo']
        mwisho= request.form['mwisho']
        date = request.form['date']
        time = request.form['time']
        amount = request.form['amount']
        conn  = pymysql.connect(host='localhost', user='root', password='', database='sierra')
        cursor=conn.cursor()
        sql = "insert into bookings(mwanzo,mwisho,date,time,amount)values(%s,%s,%s,%s,%s)"
        
        
        try:
            cursor.execute(sql,(mwanzo,mwisho,date,time,amount))
            conn.commit()
            return render_template('booking.html', msg = "Saved")
        except:
            
            return render_template('booking.html', msg = "Failed")
    else:
         return render_template('booking.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        Flour = request.form.get("Flour")
        Milk = request.form.get("Milk")
        Sugar = request.form.get("Sugar")
        Butter = request.form.get("Butter")
        Egg = request.form.get("Egg")
        BakingPowder= request.form.get("Baking Powder")
        Vanilla = request.form.get("Vanilla")
        Salt = request.form.get("Salt")

        
        # Put inputs to dataframe
        X = pandas.DataFrame([[Flour, Milk,Sugar,Butter,Egg,BakingPowder,Vanilla,Salt]], columns = ["Flour", "Milk","Sugar","Butter","Egg","Baking Powder","Vanilla","Salt"])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("model.html", output = prediction)




@app.route('/index', methods = ['GET','POST'])
def index():
    return render_template('index.html')


@app.route ('/morning',methods=['POST','GET'])
def morning():
    
        conn  = pymysql.connect(host='localhost', user='root', password='', database='sierra')
        cursor=conn.cursor()
        sql = 'select * from track'
        cursor.execute(sql)
        row = cursor.fetchone()
        tra = row[1]
        


        if tra > 29:
           return render_template('empty.html', msg = "The Bus Is Fully Booked")

        else:
            tra = tra + 1
            if request.method == 'POST':
                names = request.form['names']
                pickup= request.form['pickup']
                phone = request.form['phone']
                
                
                
                
                sql = "insert into booki(names,pickup,phone)values(%s,%s,%s)"
            
            
                try:
                    cursor.execute(sql,(names,pickup,phone))
                    #send sms
                    
                    conn.commit()
                    sql1 = 'UPDATE track SET tracking = %s'
                    cursor1 = conn.cursor()
                    cursor1.execute(sql1,(tra))
                    conn.commit()
                    

                    return render_template('booking1.html', msg = "Booking Reserved")
                except:
                    
                
                    return render_template('booking1.html', msg = "Failed")
            else:
                return render_template('booking1.html')

    

 



@app.route("/viewbookings")
def viewbookings():
    #connect the database
    conn  = pymysql.connect(host='localhost', user='root', password='', database='sierra')

    # Create cursor 
    cursor = conn.cursor()
    #prepare the SQl, selecting all records in bookings table
    sql = "SELECT * FROM bookings"
    # execute the SQL
    cursor.execute(sql)
    # check how many rows were returned, if its zero, return

    # message back to template
    if cursor.rowcount==0:
        return render_template('viewbookings.html', msg="No Records")
    else:
    # return rows found back to template
        rows = cursor.fetchall()
        return render_template('viewbookings.html', rows=rows)



@app.route("/view")
def view():
    #connect the database
    conn = pymysql.connect(host='fabemed.mysql.pythonanywhere-services.com', user='fabemed',password='farah2001',database='fabemed$default')

    # Create cursor 
    cursor = conn.cursor()
    #prepare the SQl, selecting all records in bookings table
    sql = "SELECT * FROM booki"
    # execute the SQL
    cursor.execute(sql)
    # check how many rows were returned, if its zero, return

    # message back to template
    if cursor.rowcount==0:
        return render_template('view.html', msg="No Records")
    else:
    # return rows found back to template
        rows = cursor.fetchall()
        return render_template('view.html', rows=rows)





if __name__ == '__main__':
    app.run(debug=True)
