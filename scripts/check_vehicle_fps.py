import json

MAKES = {'ford','toyota','honda','chevrolet','dodge','jeep','bmw','mercedes',
         'volkswagen','nissan','hyundai','kia','subaru','mazda','audi','volvo',
         'lexus','acura','infiniti','cadillac','buick','gmc','chrysler','ram',
         'lincoln','tesla','lamborghini','porsche','ferrari','maserati',
         'mitsubishi','pontiac','saturn','oldsmobile','mercury','hummer',
         'isuzu','suzuki','scion','fiat','jaguar','land rover','mini'}

suspicious = []
total_vehicle = 0

with open('data/rejected.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('_rejection_reason') != 'vehicle_listing':
            continue
        total_vehicle += 1
        title = r.get('title', '')
        if not any(make in title.lower() for make in MAKES):
            suspicious.append(title[:120])

print(f"Total vehicle rejections: {total_vehicle}")
print(f"Suspicious (no known make): {len(suspicious)}")
print()
for i, t in enumerate(suspicious[:30], 1):
    print(f"{i:3d} | {t}")
