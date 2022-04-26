#include <PubSubClient.h>



#include <LiquidCrystal_I2C.h> //make sure it is the library at: https://github.com/johnrickman/LiquidCrystal_I2C

#include <NTPClient.h>
#include <WiFiUdp.h>
//#include <LiquidCrystal_I2C.h>
#include <Wire.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>
#include <ESP8266WiFi.h>

const char* mqtt_server = "test.mosquitto.org";
const int mqtt_port = 1883;
WiFiClient espClient;
PubSubClient client(espClient);

static const int RXPin = 2, TXPin = 0;
static const uint32_t GPSBaud = 9600;

//WiFi details
const char* ssid = "PrimroseN"; //ssid of your wifi
const char* password = "anirudhananya"; //password of your wifi

const long utcOffsetInSeconds = 19800;

// Define NTP Client to get time
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", utcOffsetInSeconds);

// Set the LCD address to 0x27 for a 16 chars and 2 line display
LiquidCrystal_I2C lcd(0x27, 16, 2);

SoftwareSerial ss(RXPin, TXPin);
TinyGPSPlus gps;

String networkTimeHour = "";
String networkTimeMinute = "";
String gpsTimeHour = "";
String gpsTimeMinute = "";

void setup()
{
  Serial.begin(115200);
  ss.begin(GPSBaud);
  
  // initialize the LCD
  lcd.begin(16,2);
  lcd.init();

  // Turn on the blacklight and print a message.
  lcd.backlight();
//  lcd.print("Line-1");
//  lcd.setCursor(0,1);
//  delay(2000);

  //Connect to WiFi
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  lcd.print("Connecting to: ");
  lcd.setCursor(0,1);
  lcd.print(ssid);
  WiFi.begin(ssid, password); //connecting to wifi
  while (WiFi.status() != WL_CONNECTED)// while wifi not connected
  {
    delay(500);
    Serial.print("."); //print "...."
  }
  Serial.println("");
  Serial.println("WiFi connected");

  lcd.clear();
  lcd.print("Connected!");
  delay(2000);

  timeClient.begin();

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(MQTTcallback);
  while (!client.connected()) 
  {
    Serial.println("Connecting to MQTT...");
    if (client.connect("ESP8266"))
    {
      Serial.println("connected");
    }
    else
    {
      Serial.print("failed with state ");
      Serial.println(client.state());
      delay(2000);
    }
  }
  client.subscribe("tarp/flight");
}
String message;
void MQTTcallback(char* topic, byte* payload, unsigned int length) 
{
  Serial.print("Message received in topic: ");
  Serial.println(topic);
  Serial.print("Message:");
  for (int i = 0; i < length; i++) 
  {
    message = message + (char)payload[i];
  }
  Serial.print(message);
  lcd.setCursor(0,1);
  lcd.print(message);
}


void loop()
{
  client.loop();
//  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("IST Time: ");

  if (WiFi.status() != WL_CONNECTED){
    getGPSTime();
  }
  else{
    getNetworkTime();
      lcd.setCursor(0,1);
  lcd.print(message);
  }
  
  delay(100);
}


void getGPSTime(){
  while (ss.available() > 0)
    if (gps.encode(ss.read())){
      gpsTimeHour = String(gps.time.hour());
      gpsTimeMinute = String(gps.time.minute());

      int gpsTimeMinuteint = gps.time.minute();
      int gpsTimeHourint = gps.time.hour();

      if(gpsTimeMinuteint < 30){
      gpsTimeMinuteint = gpsTimeMinuteint+30;
      }
      else{
        gpsTimeMinuteint = gpsTimeMinuteint-30;
        gpsTimeHourint++;
      }

      gpsTimeHourint = gpsTimeHourint+5;
      if(gpsTimeHourint >= 24){
        gpsTimeHourint = gpsTimeHourint-24;
      }

      gpsTimeMinute = String(gpsTimeMinuteint);
      gpsTimeHour = String(gpsTimeHourint);

      if(gpsTimeHourint < 10){
        gpsTimeHour = "0" + gpsTimeHour;
      }
      if(gpsTimeMinuteint < 10){
        gpsTimeMinute = "0" + gpsTimeMinute;
      }
      lcd.setCursor(10,0);
      lcd.print(gpsTimeHour+":"+gpsTimeMinute);
    }
    lcd.setCursor(0,1);
    lcd.print("(GPS Time)    ");
}

void getNetworkTime(){
  timeClient.update();
  networkTimeHour = String(timeClient.getHours());
  networkTimeMinute = String(timeClient.getMinutes());
  if(timeClient.getHours() < 10){
    networkTimeHour = "0" + networkTimeHour;
  }
  if(timeClient.getMinutes() < 10){
    networkTimeMinute = "0" + networkTimeMinute;
  }
  lcd.setCursor(10,0);
  lcd.print(networkTimeHour+":"+networkTimeMinute);
//  if(message == ""){
//      lcd.setCursor(0,1);
//  lcd.print("(Network Time)");
//  }
}
