CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = sph_baseline
SOURCES = sph_baseline.cpp container.cpp
HEADERS = sph_baseline.h container.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) 60

clean:
	rm -f $(TARGET)
	rm -f output/frame_*.csv
