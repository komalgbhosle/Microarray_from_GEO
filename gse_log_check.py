#!/usr/bin/env python
# -*- coding: utf-8 -*-

import GEOparse
import requests
from bs4 import BeautifulSoup

def check_log(gsm_name):
	url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" +  gsm_name
	resp = requests.get(url)
	soup = BeautifulSoup(resp.content, 'lxml')
	main_data = soup.find_all('table', {"cellspacing":"3"})
	for content in main_data:
		print(content)
		if 'Data table header descriptions' in content.text and ('log' in content.text or 'log2' in content.text):
			return True

def check_gsm_data(gse_id):
	gse = GEOparse.get_GEO(geo=gse_id, destdir="./")
	for gsm_name, gsm in gse.gsms.items():
		result = check_log(gsm_name)
		if result:
			print("log file available")
		else:
			print("log file unavailable")
		break

gse_id = "GSE15783"
check_gsm_data(gse_id)