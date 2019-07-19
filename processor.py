from datetime import datetime
import mysql.connector
import pandas as pd
import smtplib
import os

import datafiller_app as dt
import processor_config as conf

# TODO: Use neutral email "info@datafiller.com"
def sendemail(from_addr, to_addr_list, cc_addr_list, bcc_addr_list,
              subject, message, login, password,
              smtpserver='smtp.gmail.com:587'):
    header = f'From: {from_addr}\n'
    header += 'To: %s\n' % ', '.join(to_addr_list)
    header += 'Cc: %s\n' % ', '.join(cc_addr_list)
    header += 'Bcc: %s\n' % ', '.join(bcc_addr_list)
    header += f'Subject: {subject}\n'
    message = header + message
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems


def get_connection():
    return mysql.connector.connect(user=conf.USER_SQL, password=conf.PWD_SQL,
                                   host=conf.HOST_SQL, database=conf.DB_SQL,
                                   connect_timeout=conf.TIMEOUT_SQL)

query_select = "SELECT * FROM works WHERE complete = 0 ORDER BY submission"

db = get_connection()
cursor = db.cursor(dictionary=True)
print(f"Executing SQL command: {query_select}")
cursor.execute(query_select)
requests = cursor.fetchall()
db.close

for request in requests:
    id = request["id"]
    input_file = request["input_file"]
    target = request["targets"]
    email = request["email"]
    print(f"Processing file: {input_file}")
    try:
        filename, file_extension = os.path.splitext(input_file)
        if file_extension == "csv":
            raw_data = pd.read_csv(conf.UPLOAD_FOLDER + input_file, low_memory=False, dtype=str)
        else:
            raw_data = pd.read_excel(conf.UPLOAD_FOLDER + input_file)
        # TODO: Restructure for filler refactoring
        filler = dt.DataFiller(raw_data, target, sparse_matrix=True)
        filler.classify_fields()
        filler.encode_fields()
        filler.predict_target()
        print("Process complete\n%s" % filler.target_predicted.head())
        output_file = conf.PROCESSED_FOLDER + filename + ".xlsx"
        filler.save_dataset(output_file)
        os.remove(conf.UPLOAD_FOLDER + input_file)
        notes = "NULL"
        msg = f"Processing successful!\n" \
              f"You can download your file at: {conf.PUBLIC_URL + output_file}\n" \
              f"The calculated field ({target}) is the leftmost column.\n\n" \
              f"Feel free to send any comment or question to: {conf.ADDRESS_EMAIL}"
        sendemail(conf.ADDRESS_EMAIL, [email], [], [conf.ADDRESS_EMAIL],
                  "DataFiller Processing Successful", msg, conf.USER_EMAIL, conf.PWD_EMAIL)

    except Exception as exception:
        print(f"Unexpected error: {exception}")
        output_file = "NULL"
        notes = str(exception).replace("\'","").replace("\"","")
        print(f"Notes: {notes}")
        msg = f"Processing Error!\nInput File: {input_file}\n" \
              f"Targets: {target}\nNotes: {notes}"
        email_result = sendemail([conf.ADDRESS_EMAIL], [conf.ADDRESS_EMAIL], [], [],
                                 "DataFiller Processing Error",
                                 msg, conf.USER_EMAIL, conf.PWD_EMAIL)

    #  database update
    query_update = "UPDATE works SET output_file = \'" + str(output_file) + \
                   "\', complete = 1, completion = \'" + str(datetime.now()) +\
                   "\', notes = \'" + notes + "\' WHERE id = " + str(id)
    print(f"Executing SQL command: {query_update}")

    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute(query_update)
    db.commit()
    db.close