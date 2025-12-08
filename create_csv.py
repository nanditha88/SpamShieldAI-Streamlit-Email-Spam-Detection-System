# create_csv.py - Generate custom email dataset CSV
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_emails(num_samples=100):
    """Generate a dataset of spam and ham emails"""
    
    spam_examples = [
        # Financial scams
        "Congratulations! You've won ${amount} {prize}! Click here to claim now.",
        "URGENT: Your {bank} account has been compromised. Verify your identity immediately.",
        "You're pre-approved for a ${amount} loan with 0% APR! Apply now.",
        "Your {credit_card} has been blocked! Click to unblock your card.",
        "Earn ${amount} weekly from home! No experience needed.",
        
        # Prize notifications
        "You've been selected for a FREE {product}! Limited time offer!",
        "Claim your ${amount} gift card! You're our lucky winner!",
        "Congratulations! You won a FREE vacation to {destination}!",
        
        # Urgent notifications
        "Your {service} account will be suspended in 24 hours!",
        "Security Alert: Unusual activity detected on your account.",
        "IMPORTANT: Your subscription is about to expire!",
        
        # Get rich quick
        "Make ${amount} in 24 hours with our proven system!",
        "Turn ${small_amount} into ${large_amount} with cryptocurrency!",
        "Become a millionaire in {timeframe}! Discover the secret!",
        
        # Health/beauty
        "Lose {weight} pounds in {days} days! Miracle supplement!",
        "Scientists discover revolutionary weight loss method!",
        "Look 10 years younger instantly! Doctors hate this trick!",
        
        # Shopping deals
        "Get 90% OFF on {product}! Limited stock!",
        "FLASH SALE: {product} for only ${price}!",
        "Black Friday deals start NOW! Up to 95% OFF!"
    ]
    
    ham_examples = [
        # Work communications
        "Hi {name}, the meeting is scheduled for {time} in {location}.",
        "Team, please review the attached {document} and provide feedback.",
        "Reminder: {event} is happening on {day} at {time}.",
        "Please submit your {report_type} by {deadline}.",
        "The {project} deadline has been extended to {new_date}.",
        
        # Personal communications
        "Hi {name}, just checking in to see how you're doing.",
        "Thanks for your email. I'll get back to you soon.",
        "Looking forward to our call tomorrow at {time}.",
        "Can we reschedule our meeting to {alternative_time}?",
        "Please find the document attached as requested.",
        
        # Notifications
        "Your order #{order_number} has been shipped. Delivery expected {date}.",
        "Your appointment with {professional} is confirmed for {date_time}.",
        "Payment of ${amount} has been received. Thank you.",
        "Your subscription to {service} has been renewed.",
        "Password reset requested for your {account} account.",
        
        # Newsletters
        "Monthly newsletter: Check out our latest updates.",
        "Company announcement: {announcement}",
        "Industry news: {summary}",
        "Weekly digest: Top stories from this week.",
        
        # Social
        "{friend_name} has accepted your connection request.",
        "You have a new message from {sender}.",
        "Event invitation: {event_name} on {date}."
    ]
    
    spam_keywords = {
        'amount': ['100', '500', '1000', '5000', '10000', '25000', '50000'],
        'prize': ['gift card', 'iPhone', 'MacBook', 'iPad', 'Samsung Galaxy', 'playstation'],
        'bank': ['PayPal', 'Bank of America', 'Chase', 'Wells Fargo', 'Citibank'],
        'credit_card': ['Visa', 'MasterCard', 'American Express', 'Discover'],
        'product': ['iPhone 15', 'Samsung S24', 'MacBook Pro', 'iPad Pro', 'PlayStation 5'],
        'destination': ['Hawaii', 'Bahamas', 'Cancun', 'Paris', 'Rome'],
        'service': ['Netflix', 'Amazon Prime', 'Spotify', 'Google', 'Microsoft'],
        'small_amount': ['10', '50', '100', '200'],
        'large_amount': ['1000', '5000', '10000', '50000'],
        'timeframe': ['30 days', '3 months', '6 months', '1 year'],
        'weight': ['10', '20', '30', '40', '50'],
        'days': ['7', '14', '21', '30'],
        'price': ['9.99', '19.99', '29.99', '49.99']
    }
    
    ham_keywords = {
        'name': ['John', 'Sarah', 'Mike', 'Emily', 'David', 'Lisa', 'Robert', 'Jennifer'],
        'time': ['9 AM', '10 AM', '2 PM', '3 PM', '4 PM'],
        'location': ['conference room A', 'main conference room', 'room 101', 'zoom meeting'],
        'document': ['report', 'proposal', 'presentation', 'document', 'file'],
        'event': ['team meeting', 'client presentation', 'training session', 'webinar'],
        'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'report_type': ['expense report', 'status report', 'weekly report', 'project report'],
        'deadline': ['EOD today', 'tomorrow', 'Friday', 'next Monday'],
        'project': ['Alpha', 'Beta', 'Gamma', 'Omega', 'Project X'],
        'new_date': ['next Friday', 'end of month', 'next week'],
        'alternative_time': ['tomorrow morning', 'Thursday afternoon', 'next week'],
        'order_number': ['12345', '67890', 'ABCDE', 'FGHIJ'],
        'date': ['tomorrow', 'in 2 days', 'next week'],
        'professional': ['Dr. Smith', 'the dentist', 'our consultant', 'the lawyer'],
        'date_time': ['tomorrow at 2 PM', 'Friday at 10 AM', 'next Monday'],
        'service': ['Netflix', 'Spotify', 'Amazon Prime', 'New York Times'],
        'account': ['Gmail', 'Facebook', 'Twitter', 'LinkedIn'],
        'announcement': ['new office opening', 'company merger', 'leadership change'],
        'summary': ['market trends', 'regulatory changes', 'technology updates'],
        'friend_name': ['Alex', 'Chris', 'Taylor', 'Jordan'],
        'sender': ['John Doe', 'Jane Smith', 'Company HR', 'Customer Support'],
        'event_name': ['Company Picnic', 'Team Building', 'Annual Conference']
    }
    
    emails = []
    labels = []
    
    # Generate spam emails
    for _ in range(num_samples // 2):
        template = random.choice(spam_examples)
        email = template
        
        # Replace placeholders with actual values
        for key, values in spam_keywords.items():
            if f"{{{key}}}" in email:
                email = email.replace(f"{{{key}}}", random.choice(values))
        
        emails.append(email)
        labels.append("spam")
    
    # Generate ham emails
    for _ in range(num_samples // 2):
        template = random.choice(ham_examples)
        email = template
        
        # Replace placeholders with actual values
        for key, values in ham_keywords.items():
            if f"{{{key}}}" in email:
                email = email.replace(f"{{{key}}}", random.choice(values))
        
        emails.append(email)
        labels.append("ham")
    
    # Shuffle the dataset
    combined = list(zip(emails, labels))
    random.shuffle(combined)
    emails, labels = zip(*combined)
    
    return pd.DataFrame({'text': emails, 'label': labels})

def save_dataset(filename="email_dataset.csv", num_samples=200):
    """Generate and save dataset to CSV"""
    df = generate_emails(num_samples)
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved as '{filename}'")
    print(f"Total emails: {len(df)}")
    print(f"Spam: {len(df[df['label'] == 'spam'])}")
    print(f"Ham: {len(df[df['label'] == 'ham'])}")
    
    # Show sample
    print("\n📧 Sample emails:")
    print(df.head(10).to_string(index=False))
    
    return df

def create_custom_csv():
    """Interactive function to create custom CSV"""
    print("📧 Email Dataset CSV Creator")
    print("=" * 40)
    
    filename = input("Enter output filename (default: email_dataset.csv): ") or "email_dataset.csv"
    num_samples = int(input("Number of samples to generate (default: 200): ") or "200")
    
    print(f"\nGenerating {num_samples} emails...")
    df = save_dataset(filename, num_samples)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"File: {filename}")
    print(f"Format: CSV with columns: text, label")
    
    return df

if __name__ == "__main__":
    create_custom_csv()