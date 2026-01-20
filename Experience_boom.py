
  VoelBoom - 5 gevoelssensoren + 5 LEDs
  -------------------------------------
  Idee:
  - 5 aanraak/druk-sensoren op verschillende plekken van je boom/plant.
  - 5 LEDs op / bij die plekken.
  - Als de user iets aanraakt, krijgt die plek een korte animatie (UX-first).
  - In rust "ademen" de LEDs zachtjes, zodat de boom leeft.

  Hardware-aanname (pas zelf aan als het anders is):
  - 5x analoge touch-/druk-sensoren (FSR, DIY druksensor, capacitieve module, etc.)
  - 5x LEDs met serieweerstand (220–330 Ω)
  - Arduino Uno / Nano

  Pin-mapping (kun je aanpassen):
  Sensoren:
    sensor 0 (blad)      -> A0
    sensor 1 (stam)      -> A1
    sensor 2 (top)       -> A2
    sensor 3 (blad lang) -> A3
    sensor 4 (potrand)   -> A4

  LEDs (PWM-pins aanbevolen):
    led 0 -> 3
    led 1 -> 5
    led 2 -> 6
    led 3 -> 9
    led 4 -> 10
*/

const int NUM_CHANNELS = 5;

// Analoge input-pins voor de gevoelssensoren
int sensorPins[NUM_CHANNELS] = {A0, A1, A2, A3, A4};

// Output-pins voor de LEDs
int ledPins[NUM_CHANNELS]    = {3, 5, 6, 9, 10};

// Drempels voor "aangeraakt"
// -> deze MOET je zelf nog kalibreren met Serial Monitor
int touchThresholds[NUM_CHANNELS] = {
  300, // kanaal 0
  300, // kanaal 1
  300, // kanaal 2
  300, // kanaal 3
  300  // kanaal 4
};

// Animatietype per kanaal (je kunt dit aanpassen)
enum AnimType {
  ANIM_NONE = 0,
  ANIM_FADE_PULSE,   // mooie fade op en neer
  ANIM_FLASH,        // snelle flitsen
  ANIM_DOUBLE_PULSE, // kort twee keer pulseren
  ANIM_SLOW_HUG      // lange langzame adem
};

AnimType channelAnimType[NUM_CHANNELS] = {
  ANIM_FADE_PULSE,   // leaf - blij bij aanraking
  ANIM_FLASH,        // stem - schrikt
  ANIM_DOUBLE_PULSE, // top - nieuwsgierig tikje
  ANIM_SLOW_HUG,     // leaf lang vasthouden-achtig
  ANIM_FADE_PULSE    // potrand - status / check
};

// Struct om state per kanaal bij te houden
struct ChannelState {
  bool          isAnimating;
  unsigned long animStart;
  unsigned long animDuration;
};

ChannelState channels[NUM_CHANNELS];

// Algemene animatieduur per type (ms)
const unsigned long DURATION_FADE_PULSE   = 2000;
const unsigned long DURATION_FLASH        = 800;
const unsigned long DURATION_DOUBLE_PULSE = 1500;
const unsigned long DURATION_SLOW_HUG     = 3000;

// Voor idle "adem" animatie
// elk kanaal krijgt een eigen offset zodat niet alles tegelijk ademt
unsigned long idleOffsets[NUM_CHANNELS] = {0, 500, 1000, 1500, 2000};

void setup() {
  Serial.begin(9600);

  // LED pins als output
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(ledPins[i], OUTPUT);
    analogWrite(ledPins[i], 0);

    channels[i].isAnimating  = false;
    channels[i].animStart    = 0;
    channels[i].animDuration = 0;
  }
}

void loop() {
  unsigned long now = millis();

  // 1. Sensoren uitlezen en animaties starten indien nodig
  for (int i = 0; i < NUM_CHANNELS; i++) {
    int sensorValue = analogRead(sensorPins[i]);

    // Debuggen? uncomment:
    // Serial.print("CH "); Serial.print(i);
    // Serial.print(" = "); Serial.println(sensorValue);

    // Een nieuwe “touch” starten alleen als we NIET al animeren
    // en de waarde boven de drempel komt.
    if (!channels[i].isAnimating && sensorValue > touchThresholds[i]) {
      startAnimation(i, now);
    }
  }

  // 2. Animaties updaten / idle gedrag tonen
  for (int i = 0; i < NUM_CHANNELS; i++) {
    if (channels[i].isAnimating) {
      updateAnimation(i, now);
    } else {
      idleGlow(i, now);
    }
  }

  // Klein delaytje om noise te verminderen en CPU wat te sparen
  delay(10);
}

// -------------------------
// Animatie-logica
// -------------------------

void startAnimation(int index, unsigned long now) {
  channels[index].isAnimating = true;
  channels[index].animStart   = now;

  // Duur kiezen op basis van animatietype
  switch (channelAnimType[index]) {
    case ANIM_FADE_PULSE:
      channels[index].animDuration = DURATION_FADE_PULSE;
      break;
    case ANIM_FLASH:
      channels[index].animDuration = DURATION_FLASH;
      break;
    case ANIM_DOUBLE_PULSE:
      channels[index].animDuration = DURATION_DOUBLE_PULSE;
      break;
    case ANIM_SLOW_HUG:
      channels[index].animDuration = DURATION_SLOW_HUG;
      break;
    default:
      channels[index].animDuration = 1000;
      break;
  }

  // Debug
  Serial.print("Start animatie kanaal ");
  Serial.print(index);
  Serial.print(" type ");
  Serial.println((int)channelAnimType[index]);
}

void updateAnimation(int index, unsigned long now) {
  unsigned long elapsed = now - channels[index].animStart;
  unsigned long duration = channels[index].animDuration;

  if (elapsed >= duration) {
    // Animatie klaar
    channels[index].isAnimating = false;
    return;
  }

  float t = (float)elapsed / (float)duration; // 0.0 - 1.0
  int brightness = 0;

  switch (channelAnimType[index]) {
    case ANIM_FADE_PULSE:
      // Smooth fade op en neer: 0 -> 1 -> 0
      if (t < 0.5f) {
        float up = t * 2.0f; // 0..1
        brightness = (int)(up * 255);
      } else {
        float down = (1.0f - t) * 2.0f; // 1..0
        brightness = (int)(down * 255);
      }
      break;

    case ANIM_FLASH: {
      // Snelle flitsen: aan-uit-aan-uit ...
      // bv. 4 flitsen
      int flashes = 4;
      float segment = 1.0f / (float)(flashes * 2); // aan + uit per flash
      int phase = (int)(t / segment); // welke fase
      // even = aan, oneven = uit
      if (phase % 2 == 0) {
        brightness = 255;
      } else {
        brightness = 0;
      }
      break;
    }

    case ANIM_DOUBLE_PULSE: {
      // Twee korte pulses in de animatie-tijd
      float localT = t * 2.0f; // twee "cycli"
      if (localT > 1.0f) localT -= 1.0f; // 0..1 binnen een cyclus

      if (localT < 0.5f) {
        float up = localT * 2.0f;
        brightness = (int)(up * 255);
      } else {
        float down = (1.0f - localT) * 2.0f;
        brightness = (int)(down * 255);
      }
      break;
    }

    case ANIM_SLOW_HUG:
      // Langzame zachte pulse, nooit volledig uit of aan
      // t in 0..1 → sin-golf
      {
        float x = t * 3.14159f; // 0..pi
        float s = (sin(x) + 0.0f) / 1.0f; // 0..1
        // tussen 40 en 200
        brightness = (int)(40 + s * 160);
      }
      break;

    default:
      brightness = 0;
      break;
  }

  brightness = constrain(brightness, 0, 255);
  analogWrite(ledPins[index], brightness);
}

// Idle "adem" animatie als er niets gebeurt
void idleGlow(int index, unsigned long now) {
  // Zacht pulseren op lage helderheid, met eigen offset per kanaal
  // zodat niet alles tegelijk gaat.

  float speed = 0.002f; // hoger = sneller ademen
  unsigned long t = now + idleOffsets[index];
  float x = (float)t * speed; // radiale tijd
  float s = (sin(x) + 1.0f) / 2.0f; // 0..1
  int maxIdle = 60;  // max brightness in idle
  int minIdle = 5;   // min brightness in idle
  int brightness = minIdle + (int)(s * (maxIdle - minIdle));

  analogWrite(ledPins[index], brightness);
}
