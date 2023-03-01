package com.flag4j.util;


import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StringUtilsTests {

    int size;
    String pad;
    String str;
    String expStr;

    @Test
    void centerTest() {
        // --------------- Sub-case 1 ---------------
        size = 13;
        str = "hello world";
        expStr = " hello world ";
        assertEquals(expStr, StringUtils.center(str, size));

        // --------------- Sub-case 2 ---------------
        size = 5;
        str = "hello world";
        expStr = "hello world";
        assertEquals(expStr, StringUtils.center(str, size));

        // --------------- Sub-case 3 ---------------
        size = 17;
        str = "hello world";
        expStr = "   hello world   ";
        assertEquals(expStr, StringUtils.center(str, size));

        // --------------- Sub-case 4 ---------------
        size = 13;
        pad = "-";
        str = "hello world";
        expStr = "-hello world-";
        assertEquals(expStr, StringUtils.center(str, size, pad));

        // --------------- Sub-case 5 ---------------
        size = 5;
        pad = "-";
        str = "hello world";
        expStr = "hello world";
        assertEquals(expStr, StringUtils.center(str, size, pad));

        // --------------- Sub-case 6 ---------------
        size = 17;
        pad = "-";
        str = "hello world";
        expStr = "---hello world---";
        assertEquals(expStr, StringUtils.center(str, size, pad));
    }
}
