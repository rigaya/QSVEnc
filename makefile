include config.mak

vpath %.cpp $(SRCDIR)

OBJS  = $(SRCS:%.cpp=%.cpp.o)
OBJCS  = $(SRCCS:%.c=%.c.o)
OBJASMS = $(ASMS:%.asm=%.o)
OBJPYWS = $(PYWS:%.pyw=%.o)
OBJRBINS = $(RBINS:%.bin=%.o)
OBJRHS = $(RHS:%.h=%.h.o)
OBJRCLS = $(RCLS:%.cl=%.o)
OBJRCLHS = $(RCLHS:%.clh=%.o)
DEPS = $(SRCS:.cpp=.cpp.d)
DEPCS = $(SRCCS:.c=.c.d)

all: $(PROGRAM)

$(PROGRAM): $(DEPS) $(DEPCS) $(OBJS) $(OBJCS) $(OBJPYWS) $(OBJRBINS) $(OBJRHS) $(OBJRCLS) $(OBJRCLHS)
	$(LD) $(OBJS) $(OBJCS) $(OBJPYWS) $(OBJRBINS) $(OBJRHS) $(OBJRCLS) $(OBJRCLHS) $(LDFLAGS) -o $(PROGRAM)

%_sse2.cpp.o: %_sse2.cpp
	$(CXX) -c $(CXXFLAGS) -msse2 -o $@ $<

%_ssse3.cpp.o: %_ssse3.cpp
	$(CXX) -c $(CXXFLAGS) -mssse3 -o $@ $<

%_sse41.cpp.o: %_sse41.cpp
	$(CXX) -c $(CXXFLAGS) -msse4.1 -o $@ $<

%_avx.cpp.o: %_avx.cpp
	$(CXX) -c $(CXXFLAGS) -mavx -mpopcnt -o $@ $<

%_avx2.cpp.o: %_avx2.cpp
	$(CXX) -c $(CXXFLAGS) -mavx2 -mpopcnt -mbmi -mbmi2 -o $@ $<

%_avx512bw.cpp.o: %_avx512bw.cpp
	$(CXX) -c $(CXXFLAGS) -mavx512f -mavx512bw -mpopcnt -mbmi -mbmi2 -o $@ $<

%.cpp.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.cpp.d: %.cpp
	@$(CXX) ./$< $(CXXFLAGS) -g0 -MT $(basename $<).cpp.o -MM > $@

%.c.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

%.c.d: %.c
	@$(CC) ./$< $(CFLAGS) -g0 -MT $(basename $<).c.o -MM > $@

%.o: %.pyw
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

%.o: %.bin
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

%.h.o: %.h
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

%.o: %.cl
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

%.o: %.clh
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

-include $(DEPS)
-include $(DEPCS)

clean:
	rm -f $(DEPS) $(DEPCS) $(OBJS) $(OBJCS) $(OBJPYWS) $(OBJRBINS) $(OBJRHS) $(OBJRCLS) $(OBJRCLHS) $(PROGRAM)

distclean: clean
	rm -f config.mak QSVPipeline/rgy_config.h

install:
	install -d $(PREFIX)/bin
	install -m 755 $(PROGRAM) $(PREFIX)/bin

uninstall:
	rm -f $(PREFIX)/bin/$(PROGRAM)

config.mak:
	./configure
