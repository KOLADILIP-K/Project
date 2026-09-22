# Check out the live app here: [drowning-incident-detection](https://drowning-incident-detection.streamlit.app/)

![](https://github.com/arrafi-musabbir/drowning-incident-detection/blob/main/images/img1_det.jpg)
### preliminary requirements (to run from Windows-10/11):
* install `python3.10` from Microsoft-store

step1: fork this repository on your account

step2: clone/download the forked repository on your PC

step3: open the cloned/downloaded directory

step4: open the terminal on your directory and create a virtual environment with the following command

    python -m venv venv

then,

    venv\Scripts\activate

step4: install all the required libraries with the following command:

    pip install -r installs.txt

step5: run `run.py` as a Python file

    python run.py
or, run `run.py` from your python-IDE (e.g. VS-CODE)

###### WhatsApp Alerts Setup
**Option 1: WAHA (Recommended - 100% Free & Self-Hosted)**
No paid API accounts or per-message fees required! WAHA runs locally via Docker:
1. Start WAHA container:
   ```bash
   docker run -d --name waha -p 3000:3000 -v waha_sessions:/app/.sessions devlikeapro/waha
   ```
2. Open the dashboard at [http://localhost:3000/dashboard](http://localhost:3000/dashboard) in your browser.
3. Scan the QR code using WhatsApp on your phone (Linked Devices).
4. Test the connection:
   ```bash
   python waha_test.py --to <YOUR_PHONE_NUMBER>
   ```

**Option 2: Twilio (Paid)**
- Check out this for Twilio account and API setup:
  [https://www.youtube.com/watch?v=O2PB6o2E8aA](https://www.youtube.com/watch?v=O2PB6o2E8aA)
- ImgBB API setup (for image hosting):
  [https://api.imgbb.com/](https://api.imgbb.com/)