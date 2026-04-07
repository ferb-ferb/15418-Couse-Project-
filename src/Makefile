CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = sph_baseline
SRC = sph_baseline.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) frame_*.csv
