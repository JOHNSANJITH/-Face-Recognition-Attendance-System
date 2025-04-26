import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Step 1: Setup scope and creds
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# Step 2: Open the sheet
sheet = client.open('AttendanceSheet').sheet1

# Step 3: Upload some dummy data
df = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Time': ['10:00:00', '10:01:00'],
    'Date': ['2025-04-08', '2025-04-08']
})

sheet.clear()  # Clear old data
sheet.append_row(df.columns.tolist())  # Add headers
for row in df.values.tolist():
    sheet.append_row(row)  # Add each row

print("✅ Upload done!")
