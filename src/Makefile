CXX = nvcc
ARCH = -arch=sm_75
CXXFLAGS = -O3 -std=c++17 $(ARCH) -ccbin g++-11 -Xcompiler "-Wall -Wextra"

TARGET = sph_baseline
SOURCES = sph_baseline.cu container.cu
HEADERS = sph_baseline.h container.h
LIBS = -lcudart

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

run: $(TARGET)
	./$(TARGET) 60

clean:
	rm -f $(TARGET)
	rm -f output/frame_*.csv
