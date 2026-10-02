FROM alpine:3.24

RUN apk add --no-cache tor curl lyrebird

RUN mkdir -p /var/lib/tor \
    && chown -R tor:tor /var/lib/tor

USER tor

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD curl --fail --max-time 10 \
    --socks5-hostname 127.0.0.1:9150 \
    https://check.torproject.org/api/ip \
    | grep -q '"IsTor":true' || exit 1

CMD ["tor", "-f", "/etc/tor/torrc"]
