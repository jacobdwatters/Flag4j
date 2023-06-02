package com.flag4j.ctensor;

import com.flag4j.CTensor;
import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CTensorToStringTests {
    static Shape aShape;
    static CNumber[] aEntries;
    static CTensor A;
    static String exp;

    @BeforeAll
    static void setup() {
        aShape = new Shape(2, 3, 1, 2);
        aEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(235.61, 1.45), new CNumber(0, -0.00024),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(1.345, -781.2),
                new CNumber(0.014, -2.45),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(51.0, 24.56),  new CNumber(6.1, -0.03),  new CNumber(-0.00014, 1.34),};
        A = new CTensor(aShape, aEntries);
    }


    @AfterAll
    static void cleanup() {
        PrintOptions.resetAll();
    }


    @Test
    void toStringTest() {
        PrintOptions.resetAll();
        // ---------------------- Sub-case 1 ----------------------
        exp = "Full Shape: 2x3x1x2\r\n" +
                "[ 1.4415 - 0.0245i  235.61 + 1.45i  -2.4E-4i  1  -85.1 + 9.234i  1.345 " +
                "- 781.2i  0.014 - 2.45i  -140  1.5i  ...  -1.4E-4 + 1.34i ]";
        assertEquals(exp, A.toString());

        // ---------------------- Sub-case 2 ----------------------
        PrintOptions.setMaxColumns(15);
        PrintOptions.setPrecision(2);
        PrintOptions.setCentering(false);
        exp = "Full Shape: 2x3x1x2\r\n" +
                "[1.44 - 0.02i  235.61 + 1.45i  0  1  -85.1 + 9.23i  1.35 - 781.2i  0.01 - 2.45i  " +
                "-140  1.5i  51 + 24.56i  6.1 - 0.03i  1.34i  ]";
        assertEquals(exp, A.toString());
    }
}
