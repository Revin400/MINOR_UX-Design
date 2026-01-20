/*
  VoelBoom - 5 gevoelskanalen + 5 LEDs (Wokwi versie)

  In Wokwi gebruik je:
  - 5 potentiometers als "aanraak/druk"-sensoren:
      linker pin -> GND
      rechter pin -> 5V
      middelste pin -> A0..A4
  - 5 LEDs:
      korte poot -> GND
      lange poot -> (optioneel via weerstand) -> pins 3,5,6,9,10

  Door aan de potmeters te draaien simuleer je aanraking/druk.
*/

const int NUM_CHANNELS = 5;

// Analoge input-pins voor de gevoelssensoren
int sensorPins[NUM_CHANNELS] = {A0, A1, A2, A3, A4};

// Output-pins voor de LEDs (PWM)
int ledPins[NUM_CHANNELS]    = {3, 5, 6, 9, 10};

// Drempels voor "aangeraakt"
// In Wokwi geven de potentiometers waardes tussen 0-1023.
// Draai je boven ongeveer midden -> "aanraking".
int touchThresholds[NUM_CHANNELS] = {
  600, // kanaal 0
  600, // kanaal 1
  600, // kanaal 2
  600, // kanaal 3
  600  // kanaal 4
};

// Types animatie per kanaal
enum AnimType {
  ANIM_NONE = 0,
  ANIM_FADE_PULSE,   // mooie fade op en neer
  ANIM_FLASH,        // snelle flitsen
  ANIM_DOUBLE_PULSE, // twee pulses
  ANIM_SLOW_HUG      // lange, zachte adem
};

AnimType channelAnimType[NUM_CHANNELS] = {
  ANIM_FADE_PULSE,   // kanaal 0
  ANIM_FLASH,        // kanaal 1
  ANIM_DOUBLE_PULSE, // kanaal 2
  ANIM_SLOW_HUG,     // kanaal 3
  ANIM_FADE_PULSE    // kanaal 4
};

// State per kanaal
struct ChannelState {
  bool          isAnimating;
  unsigned long animStart;
  unsigned long animDuration;
};

ChannelState channels[NUM_CHANNELS];

// Durations (ms)
const unsigned long DURATION_FADE_PULSE   = 2000;
const unsigned long DURATION_FLASH        = 800;
const unsigned long DURATION_DOUBLE_PULSE = 1500;
const unsigned long DURATION_SLOW_HUG     = 3000;

// Idle offsets zodat niet alle LEDs precies tegelijk ademen
unsigned long idleOffsets[NUM_CHANNELS] = {0, 500, 1000, 1500, 2000};

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(ledPins[i], OUTPUT);
    analogWrite(ledPins[i], 0);

    channels[i].isAnimating  = false;
    channels[i].animStart    = 0;
    channels[i].animDuration = 0;
  }

  Serial.println("VoelBoom Wokwi simulator gestart!");
}

void loop() {
  unsigned long now = millis();

  // 1. Sensoren uitlezen en animaties starten indien nodig
  for (int i = 0; i < NUM_CHANNELS; i++) {
    int sensorValue = analogRead(sensorPins[i]);

    // Debuggen: laat waardes zien in Serial Monitor
    // zodat je drempels kunt finetunen
    // Serial.print("CH ");
    // Serial.print(i);
    // Serial.print(" = ");
    // Serial.println(sensorValue);

    if (!channels[i].isAnimating && sensorValue > touchThresholds[i]) {
      startAnimation(i, now);
    }
  }

  // 2. Animaties of idle-gedrag updaten
  for (int i = 0; i < NUM_CHANNELS; i++) {
    if (channels[i].isAnimating) {
      updateAnimation(i, now);
    } else {
      idleGlow(i, now);
    }
  }

  delay(10); // een klein beetje rust
}

// -------------------------
// Animatie-logica
// -------------------------

void startAnimation(int index, unsigned long now) {
  channels[index].isAnimating = true;
  channels[index].animStart   = now;

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

  Serial.print("Start animatie kanaal ");
  Serial.print(index);
  Serial.print(" (type ");
  Serial.print((int)channelAnimType[index]);
  Serial.println(")");
}

void updateAnimation(int index, unsigned long now) {
  unsigned long elapsed  = now - channels[index].animStart;
  unsigned long duration = channels[index].animDuration;

  if (elapsed >= duration) {
    channels[index].isAnimating = false;
    return;
  }

  float t = (float)elapsed / (float)duration; // 0..1
  int brightness = 0;

  switch (channelAnimType[index]) {
    case ANIM_FADE_PULSE:
      // 0 -> 1 -> 0
      if (t < 0.5f) {
        float up = t * 2.0f; // 0..1
        brightness = (int)(up * 255);
      } else {
        float down = (1.0f - t) * 2.0f; // 1..0
        brightness = (int)(down * 255);
      }
      break;

    case ANIM_FLASH: {
      // 4 flitsen: aan-uit-aan-uit...
      int flashes = 4;
      float segment = 1.0f / (float)(flashes * 2); // 8 segmenten
      int phase = (int)(t / segment);
      if (phase % 2 == 0) {
        brightness = 255; // aan
      } else {
        brightness = 0;   // uit
      }
      break;
    }

    case ANIM_DOUBLE_PULSE: {
      // Twee pulses in de animatie-tijd
      float cycles = 2.0f;
      float localT = t * cycles; // 0..2
      if (localT > 1.0f) localT -= 1.0f; // breng terug naar 0..1

      if (localT < 0.5f) {
        float up = localT * 2.0f;
        brightness = (int)(up * 255);
      } else {
        float down = (1.0f - localT) * 2.0f;
        brightness = (int)(down * 255);
      }
      break;
    }

    case ANIM_SLOW_HUG: {
      // Langzame, zachte golf: nooit helemaal uit of max
      float x = t * 3.14159f; // 0..pi
      float s = (sin(x) + 0.0f) / 1.0f; // 0..1
      int minB = 40;
      int maxB = 200;
      brightness = minB + (int)(s * (maxB - minB));
      break;
    }

    default:
      brightness = 0;
      break;
  }

  brightness = constrain(brightness, 0, 255);
  analogWrite(ledPins[index], brightness);
}

// Idle "adem" animatie als er niets gebeurt
void idleGlow(int index, unsigned long now) {
  float speed = 0.002f;
  unsigned long t = now + idleOffsets[index];
  float x = (float)t * speed;
  float s = (sin(x) + 1.0f) / 2.0f; // 0..1

  int maxIdle = 60;
  int minIdle = 5;
  int brightness = minIdle + (int)(s * (maxIdle - minIdle));

  analogWrite(ledPins[index], brightness);
}
