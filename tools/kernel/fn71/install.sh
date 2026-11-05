#!/usr/bin/env bash
# FN71 install -- THE USER RUNS THIS AS ROOT (agents have no root and never try).
#
#   sudo FN71_P2PMEM_MB=2048 bash tools/kernel/fn71/install.sh      # then reboot into 7.1.3-p2p
#
# Installs the patched, signed amdgpu.ko (tools/kernel/fn71/0001-*.patch, built by
# build_module.sh in ~/kbuild/linux-7.1.3) for kernel 7.1.3-p2p ONLY:
#   1. backs up the running module to amdgpu.ko.fn70-orig (once; never overwritten),
#   2. installs the new .ko at the same path + depmod,
#   3. rebuilds the 7.1.3-p2p initramfs ONLY if it carries amdgpu (backup .fn70-orig first),
#   4. installs a udev rule + helper that hands /sys/bus/pci/devices/<gpu>/p2pmem/allocate to the
#      user (uid FN71_UID, default 1000) at every boot -- the file is root 0660 and the
#      rootless container bind-mounts it,
#   5. adds amdgpu.p2pmem_mb=N to the 7.1.3-p2p boot entry ONLY (grubby); the Fedora 7.0.11 entry is
#      untouched and stays the fallback boot.
# Prints the rollback at the end. Idempotent: a second run re-installs the same .ko and leaves the
# backups alone.
set -euo pipefail
KVER=7.1.3-p2p
N=${FN71_P2PMEM_MB:-2048}
UIDN=${FN71_UID:-1000}
KTREE=${FN71_KTREE:-~/kbuild/linux-7.1.3}
SRC=$KTREE/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
MODDIR=/lib/modules/$KVER/kernel/drivers/gpu/drm/amd/amdgpu
DST=$MODDIR/amdgpu.ko
INITRD=/boot/initramfs-$KVER.img
VMLINUZ=/boot/vmlinuz-$KVER

[ "$(id -u)" = 0 ] || { echo "fn71 install: run as root (sudo)" >&2; exit 1; }
case "$N" in ''|*[!0-9]*) echo "FN71_P2PMEM_MB must be a number of MiB" >&2; exit 1 ;; esac
[ -f "$SRC" ] || { echo "no built module at $SRC (run build_module.sh first)" >&2; exit 1; }
[ -f "$DST" ] || { echo "no installed module at $DST" >&2; exit 1; }
[ -f "$VMLINUZ" ] || { echo "no $VMLINUZ" >&2; exit 1; }

# the module must be THIS kernel's and carry the parameter
vm=$(modinfo -F vermagic "$SRC")
case "$vm" in "$KVER "*) ;; *) echo "vermagic '$vm' is not $KVER" >&2; exit 1 ;; esac
modinfo -F parm "$SRC" | grep -q '^p2pmem_mb:' || { echo "$SRC has no p2pmem_mb parameter" >&2; exit 1; }
echo "== module: $SRC"
echo "   vermagic $vm"
echo "   sha256   $(sha256sum "$SRC" | cut -d' ' -f1)"
echo "   signer   $(modinfo -F signer "$SRC" || true)"

# 1-2. back up once, install, depmod
if [ ! -e "$DST.fn70-orig" ]; then
  cp -p "$DST" "$DST.fn70-orig"
  echo "== backed up $DST -> $DST.fn70-orig"
else
  echo "== backup $DST.fn70-orig exists (kept)"
fi
install -m 0644 -o root -g root "$SRC" "$DST"
restorecon "$DST" 2>/dev/null || true
depmod -a "$KVER"
echo "== installed + depmod -a $KVER"

# 3. initramfs: ALWAYS rebuilt (FN71b, 2026-10-08: on boot 1 the lsinitrd check did not trigger
#    `dracut -f` -- the image carried amdgpu but the check missed it (lsinitrd's listing / the
#    compressed image), so the OLD amdgpu.ko loaded from the initramfs and p2pmem_mb was absent;
#    the user rebuilt by hand and rebooted). A rebuild is cheap and never wrong; then CHECK the
#    result carries the new module (same size as the installed .ko) when it carries amdgpu at all.
[ -e "$INITRD.fn70-orig" ] || cp -p "$INITRD" "$INITRD.fn70-orig"
echo "== dracut -f $INITRD $KVER (backup $INITRD.fn70-orig)"
dracut -f "$INITRD" "$KVER"
if lsinitrd "$INITRD" 2>/dev/null | grep -q 'amdgpu\.ko'; then
  echo "== $INITRD carries amdgpu.ko (rebuilt from the installed module)"
  lsinitrd "$INITRD" | grep 'amdgpu\.ko'
else
  echo "== $INITRD does not carry amdgpu: the root filesystem's module loads"
fi

# 4. hand the p2pmem allocate file to the user at every boot
cat > /usr/local/sbin/fn71-p2pmem-perms <<EOF
#!/bin/sh
# FN71: give uid $UIDN the PCI P2PDMA pool file of an amdgpu device (called by udev with %k).
f=/sys/bus/pci/devices/\$1/p2pmem/allocate
[ -e "\$f" ] || exit 0
chown $UIDN "\$f" && chmod 0660 "\$f"
EOF
chmod 0755 /usr/local/sbin/fn71-p2pmem-perms
cat > /etc/udev/rules.d/71-fn71-p2pmem.rules <<'EOF'
# FN71: the p2pmem pool file appears when amdgpu (amdgpu.p2pmem_mb > 0) binds; coldplug sends "add"
ACTION=="add|bind", SUBSYSTEM=="pci", DRIVER=="amdgpu", RUN+="/usr/local/sbin/fn71-p2pmem-perms %k"
EOF
udevadm control --reload 2>/dev/null || true
echo "== udev rule /etc/udev/rules.d/71-fn71-p2pmem.rules + /usr/local/sbin/fn71-p2pmem-perms (uid $UIDN)"

# 5. the cmdline, 7.1.3-p2p only (replace any earlier value)
grubby --update-kernel="$VMLINUZ" --remove-args="amdgpu.p2pmem_mb" >/dev/null 2>&1 || true
if [ "$N" != 0 ]; then
  grubby --update-kernel="$VMLINUZ" --args="amdgpu.p2pmem_mb=$N"
fi
echo "== boot entry for $KVER:"
grubby --info="$VMLINUZ" | grep -E '^(kernel|args)='

cat <<EOF

== DONE. Reboot into $KVER. After the boot, as your user:
     cat /sys/module/amdgpu/parameters/p2pmem_mb            # $N
     journalctl -k -b | grep p2pmem                          # 2 lines, one per R9700
     cat /sys/bus/pci/devices/0000:0{3,7}:00.0/p2pmem/{size,available,published}
   then the hold: bash tools/kernel/fn71/verify.sh

== ROLLBACK (as root), then reboot:
     cp -p $DST.fn70-orig $DST && depmod -a $KVER
     [ -e $INITRD.fn70-orig ] && cp -p $INITRD.fn70-orig $INITRD
     grubby --update-kernel=$VMLINUZ --remove-args=amdgpu.p2pmem_mb
     rm -f /etc/udev/rules.d/71-fn71-p2pmem.rules /usr/local/sbin/fn71-p2pmem-perms
   Quick off without reinstalling: boot with amdgpu.p2pmem_mb=0 (edit the entry in GRUB with 'e'),
   the patched module then does nothing. If $KVER does not come up at all: pick the Fedora 7.0.11
   entry in the GRUB menu and run the rollback from there.
EOF
