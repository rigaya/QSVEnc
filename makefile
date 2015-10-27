include config.mak

vpath %.cpp $(SRCDIR)
vpath %.asm $(SRCDIR)

OBJS  = $(SRCS:%.cpp=%.o)
OBJASMS = $(ASMS:%.asm=%.o)

all: $(PROGRAM)

$(PROGRAM): .depend $(OBJS) $(OBJASMS)
	$(LD) $(OBJS) $(OBJASMS) $(LDFLAGS) -o $(PROGRAM)

%.o: %.cpp .depend
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.o: %.asm
	$(AS) $(ASFLAGS) -o $@ $<
.depend: config.mak
	@rm -f .depend
	@echo 'generate .depend...'
	@$(foreach SRC, $(SRCS:%=$(SRCDIR)/%), $(CXX) $(SRC) $(CXXFLAGS) -g0 -MT $(SRC:$(SRCDIR)/%.cpp=%.o) -MM >> .depend;)
	
ifneq ($(wildcard .depend),)
include .depend
endif

clean:
	rm -f $(OBJS) $(OBJASMS) $(PROGRAM) .depend

distclean: clean
	rm -f config.mak

install:
	install -d $(PREFIX)/bin
	install -m 755 $(PROGRAM) $(PREFIX)/bin

uninstall:
	rm -f $(PREFIX)/bin/$(PROGRAM)

config.mak:
	./configure
