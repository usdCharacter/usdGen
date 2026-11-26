// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_FLOAT32_ENVELOPE_GLSL
#define USDGEN_FLOAT32_ENVELOPE_GLSL
// IEEE binary32 round-to-nearest/even arithmetic for the final envelope.
// Vulkan devices may flush subnormals and need not support DenormPreserve32.
// Keep these values as integer bits through subtraction, multiply and addition.
uint shiftJam(uint x,uint n) {
    if(n==0u) return x;
    if(n>=32u) return uint(x!=0u);
    return (x>>n)|uint((x&((1u<<n)-1u))!=0u);
}
uint roundPack(uint sign,int exponent,uint significand) {
    if(significand==0u) return sign;
    if(exponent < -126) {
        significand=shiftJam(significand,uint(-126-exponent));
        exponent=-126;
    }
    uint tail=significand&7u;
    uint mantissa=(significand>>3u)+uint(tail>4u||(tail==4u&&((significand>>3u)&1u)!=0u));
    if(mantissa>=0x1000000u) { mantissa>>=1u; ++exponent; }
    if(exponent>127) return sign|0x7f800000u;
    uint field=mantissa>=0x800000u?uint(exponent+127):0u;
    return sign|(field<<23u)|(mantissa&0x7fffffu);
}
void unpackFinite(uint bits,out int exponent,out uint significand) {
    uint field=(bits>>23u)&255u;
    exponent=field==0u?-126:int(field)-127;
    significand=(bits&0x7fffffu)|(field==0u?0u:0x800000u);
    if(significand!=0u) while(significand<0x800000u) { significand<<=1u; --exponent; }
}
uint addBits(uint a,uint b) {
    uint ma=a&0x7fffffffu,mb=b&0x7fffffffu;
    if(ma>=0x7f800000u||mb>=0x7f800000u) {
        if(ma>0x7f800000u||mb>0x7f800000u||
           (ma==0x7f800000u&&mb==0x7f800000u&&((a^b)&0x80000000u)!=0u)) return 0x7fc00000u;
        return ma==0x7f800000u?a:b;
    }
    if(ma==0u&&mb==0u) return (a&b)&0x80000000u;
    if(ma==0u) return b;
    if(mb==0u) return a;
    // Larger absolute operand determines sign and alignment.
    if(ma<mb) { uint t=a;a=b;b=t; }
    int ea,eb;uint sa,sb;unpackFinite(a,ea,sa);unpackFinite(b,eb,sb);
    sa<<=3u;sb=shiftJam(sb<<3u,uint(ea-eb));
    uint sign=a&0x80000000u;
    if(((a^b)&0x80000000u)==0u) {
        sa+=sb;
        if(sa>=0x8000000u) { sa=shiftJam(sa,1u);++ea; }
    } else {
        sa-=sb;
        if(sa==0u) return 0u;
        while(sa<0x4000000u) { sa<<=1u;--ea; }
    }
    return roundPack(sign,ea,sa);
}
uint multiplyBits(uint a,uint b) {
    uint sign=(a^b)&0x80000000u,ma=a&0x7fffffffu,mb=b&0x7fffffffu;
    if(ma>=0x7f800000u||mb>=0x7f800000u) {
        if(ma>0x7f800000u||mb>0x7f800000u||ma==0u||mb==0u) return 0x7fc00000u;
        return sign|0x7f800000u;
    }
    if(ma==0u||mb==0u) return sign;
    int ea,eb;uint sa,sb;unpackFinite(a,ea,sa);unpackFinite(b,eb,sb);
    uint hi,lo;umulExtended(sa,sb,hi,lo);
    bool upper=(hi&0x8000u)!=0u;
    uint shift=upper?21u:20u;
    uint significand=(hi<<(32u-shift))|(lo>>shift);
    significand|=uint((lo&((1u<<shift)-1u))!=0u);
    return roundPack(sign,ea+eb+(upper?1:0),significand);
}
float blendPoint(float original,float changed,uint envelopeBits) {
    uint a=floatBitsToUint(original),b=floatBitsToUint(changed);
    uint delta=addBits(b,a^0x80000000u);
    return uintBitsToFloat(addBits(a,multiplyBits(delta,envelopeBits)));
}
#endif
