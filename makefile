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

all: $(PROGRAM)

$(PROGRAM): .depend $(OBJS) $(OBJCS) $(OBJPYWS) $(OBJRBINS) $(OBJRHS) $(OBJRCLS) $(OBJRCLHS)
	$(LD) $(OBJS) $(OBJCS) $(OBJPYWS) $(OBJRBINS) $(OBJRHS) $(OBJRCLS) $(OBJRCLHS) $(LDFLAGS) -o $(PROGRAM)

%_sse2.cpp.o: %_sse2.cpp .depend
	$(CXX) -c $(CXXFLAGS) -msse2 -o $@ $<

%_ssse3.cpp.o: %_ssse3.cpp .depend
	$(CXX) -c $(CXXFLAGS) -mssse3 -o $@ $<

%_sse41.cpp.o: %_sse41.cpp .depend
	$(CXX) -c $(CXXFLAGS) -msse4.1 -o $@ $<

%_avx.cpp.o: %_avx.cpp .depend
	$(CXX) -c $(CXXFLAGS) -mavx -o $@ $<

%_avx2.cpp.o: %_avx2.cpp .depend
	$(CXX) -c $(CXXFLAGS) -mavx2 -o $@ $<

%.cpp.o: %.cpp .depend
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.c.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

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
	
.depend: config.mak
	@rm -f .depend
	@echo 'generate .depend...'
	@$(foreach SRC, $(SRCS:%=$(SRCDIR)/%), $(CXX) $(SRC) $(CXXFLAGS) -g0 -MT $(SRC:$(SRCDIR)/%.cpp=%.o) -MM >> .depend;)
	
ifneq ($(wildcard .depend),)
include .depend
endif

clean:
	rm -f $(OBJS) $(OBJCS) $(OBJPYWS) $(OBJRBINS) $(OBJRHS) $(OBJRCLS) $(OBJRCLHS) $(PROGRAM) .depend

distclean: clean
	rm -f config.mak QSVPipeline/qsv_config.h

install:
	install -d $(PREFIX)/bin
	install -m 755 $(PROGRAM) $(PREFIX)/bin

uninstall:
	rm -f $(PREFIX)/bin/$(PROGRAM)

config.mak:
	./configure
