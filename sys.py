
from django.shortcuts import render
from flask import *

import pymysql

app =Flask( __name__)
app.secret_key='!$F./rE$4T*_8.Vc#@f5'






@app.route('/book',methods=['GET','POST'])
def book():
    if 'phone' in session:

        conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
        cursor = conn.cursor()

        if request.method == 'POST':
            mwanzo = request.form['mwanzo']
            mwisho = request.form['mwisho']
            date = request.form['date']
            time = request.form['time']
            amount = request.form['amount']
            phone=session['phone']

            sql = 'insert into bookings(mwanzo,mwisho,date,time,amount,phone)values(%s,%s,%s,%s,%s,%s)'
            cursor.execute(sql,(mwanzo,mwisho,date,time,amount,phone))
            conn.commit()
            return render_template('booking.html', msg='Booking Reserved')
        else:
            return render_template('booking.html')
    else:
        return redirect('/login')

@app.route('/checkout/<booking_id>')
def checkout(booking_id):
    if 'phone' in session:
        conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
        cursor = conn.cursor()
        sql = 'delete from bookings where booking_id=%s'
        cursor.execute(sql,(booking_id))
        conn.commit()
        # track
       
        return redirect('/viewbookings')
    else:
        return redirect('/login')









@app.route('/signup',methods=['GET','POST'])
def signup():
    #Create a connection
    conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
    cursor = conn.cursor()
    #If someone has posted something in the form
    if request.method == 'POST':
        names = request.form['names']
        email = request.form['email']
        password = request.form['password']
        phone = request.form['phone']
        gender = request.form['gender']


        if len(password) <8:
            return render_template('home1.html',msg='your password should be more than 8 characters')
        else:
            sql = 'insert into user(names,email,password,phone,gender)values(%s,%s,%s,%s,%s)'
            cursor.execute(sql,(names,email,password,phone,gender))
            conn.commit()
            return render_template('login.html', msg='Saved successfully')
    else:
        return render_template('home1.html')


@app.route('/login',methods=['GET','POST'])
def login():
    conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
    cursor = conn.cursor()

    if request.method == 'POST':
        phone = request.form['phone']
        password= request.form['password']


        sql = 'select * from user where phone=%s and password=%s'
        cursor.execute(sql,(phone,password))
        if cursor.rowcount ==0:
            return render_template('login.html',msg='Invalid Credentials')
        else:
            row = cursor.fetchone()
            session['phone']=row[3]
            
            return render_template('index.html')
    else:
        return render_template('login.html')


@app.route('/')
def index():
    if 'phone' in session:
         return render_template('index.html')
    else:
        return redirect('/login')

@app.route('/viewbookings')
def viewbookings():
    if 'phone' in session:
        
        conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
        cursor = conn.cursor()
        session_key = session['phone']

        sql = 'select * from bookings where phone=%s'
        cursor.execute(sql,(session_key))
        if cursor.rowcount == 0:
            return render_template('viewbookings.html',msg='No Bookings Available')
        else:
            rows=cursor.fetchall()
            return render_template('viewbookings.html',rows=rows)
    else:
        return redirect('/login')



@app.route('/hire')
def hire():
    if 'phone' in session:
        conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
        cursor = conn.cursor()
        sql = 'select * from cars where status="available"   '
        cursor.execute(sql)
        if cursor.rowcount == 0:
            return render_template('hire.html', msg='No cars available')
        else:
            rows=cursor.fetchall()
            return render_template('hire.html', rows=rows)
    else:
        return redirect('/login')



import requests
import datetime
import base64
from requests.auth import HTTPBasicAuth
@app.route('/mpesa', methods = ['POST','GET'])
def mpesa():
        if request.method == 'POST':
            phone = str(request.form['phone'])
            amount = str(request.form['amount'])
            # GENERATING THE ACCESS TOKEN
            consumer_key = "GTWADFxIpUfDoNikNGqq1C3023evM6UH"
            consumer_secret = "amFbAoUByPV2rM5A"

            api_URL = "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials" #AUTH URL
            r = requests.get(api_URL, auth=HTTPBasicAuth(consumer_key, consumer_secret))

            data = r.json()
            access_token = "Bearer" + ' ' + data['access_token']

            #  GETTING THE PASSWORD
            timestamp = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
            passkey = 'bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919'
            business_short_code = "174379"
            data = business_short_code + passkey + timestamp
            encoded = base64.b64encode(data.encode())
            password = encoded.decode('utf-8')


            # BODY OR PAYLOAD
            payload = {
                "BusinessShortCode": "174379",
                "Password": "{}".format(password),
                "Timestamp": "{}".format(timestamp),
                "TransactionType": "CustomerPayBillOnline",
                "Amount": '1',  # use 1 when testing
                "PartyA": phone,  # change to your number
                "PartyB": "174379",
                "PhoneNumber": phone,
                "CallBackURL": "https://modcom.co.ke/job/confirmation.php",
                "AccountReference": "account",
                "TransactionDesc": "account"
            }

            # POPULAING THE HTTP HEADER
            headers = {
                "Authorization": access_token,
                "Content-Type": "application/json"
            }

            url = "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest" #C2B URL

            response = requests.post(url, json=payload, headers=headers)
            print (response.text)
            return ('Complete Your Payment On your Phone')
        else:
            return render_template('single.html')










@app.route('/single/<reg>')
def single(reg):
    conn = pymysql.connect(host='localhost',user='root',password='',database='sierra')
    cursor = conn.cursor()
    sql='select * from cars where reg=%s'
    cursor.execute(sql,(reg))
    row = cursor.fetchone()
    return render_template('single.html', row=row)




   


@app.route('/save',methods=['POST','GET'])
def save():
    conn = pymysql.connect(host='localhost',user='root',password='',database='digg')
    cursor = conn.cursor()

    if request.method == 'POST':
        name = request.form['names']
        email = request.form['email']



        sql = 'insert into user_table(names,email,password,phone,gender)values(%s,%s,%s,%s,%s)'
        cursor.execute(sql,(name,))
        conn.commit()
        return render_template('save1.html',msg='Saved successfully')

    else:
        return render_template('hire.html')
        
        










    




       
@app.route('/signout')
def signout():
   session.pop('key',None)
   return redirect('/login')
        
    







if __name__ == '__main__':
    app.run(debug=True)