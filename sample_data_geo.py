#!/usr/bin/env python
# -*- coding: utf-8 -*-


import requests
from bs4 import BeautifulSoup

def parse_data(accession_id):
	url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + accession_id
	resp = requests.get(url)
	soup = BeautifulSoup(resp.content, 'lxml')
	all_rows = soup.find_all('tr', {"valign":"top"})
	insert_dict = {}
	for ele in all_rows:
		main_text = ele.find('td').text
		if main_text == "BioProject":
			key_data = ele.find_all('td')[1].text
			print(key_data)


accession_id = "GSE133059"
parse_data(accession_id)