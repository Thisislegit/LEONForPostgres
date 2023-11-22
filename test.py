import re
json_msg = text='{"Plans": [{"Node Type": "SeqScan","Node Type ID": "169","Relation IDs": "kt","Base Restrict Info": "kt.kind::text = ANY (\'{movie,"tv movie","video movie","video game"}\'::text[])"}, ,"wyz": "ci.note = ANY (\'{(writer),"(head writer)","(written by)",(story),"(story editor)"}\'::text[])"}'
def fix_json_msg(json):
    # pattern = r'ANY \((.*?)\)'
    pattern = r'ANY \((.*?):text\[\]\)'
    matches = re.findall(pattern, json)
    for match in matches:
        print(match)
        extracted_string = match
        cleaned_string = extracted_string.replace('"', '')
        json = json.replace(extracted_string, cleaned_string)
    return json
json_msg = fix_json_msg(json_msg)
print(json_msg)