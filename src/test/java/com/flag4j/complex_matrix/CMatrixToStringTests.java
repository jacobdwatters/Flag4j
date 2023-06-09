package com.flag4j.complex_matrix;

import com.flag4j.CMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.io.PrintOptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixToStringTests {

    CNumber[][] aEntries;
    CMatrix A;
    String exp;


    @Test
    void toStringTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(50);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "[[    2 + 4.25i     -2.345E-4    9.4i         1.2598       ]\n" +
                " [ 56.25 - 0.0024i      0      -1.4545i  -3.356 - 84.2525i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(3);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "[[   2 + 4.25i     0   9.4i          1.26       ]\n" +
                " [ 56.25 - 0.002i  0  -1.455i  -3.356 - 84.253i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 3 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(0);
        PrintOptions.setPadding(4);
        PrintOptions.setMaxRows(50);
        PrintOptions.setMaxColumns(50);
        PrintOptions.setCentering(true);
        exp = "[[  2 + 4i    0    9i       1      ]\n" +
                " [    56      0    -i    -3 - 84i  ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 4 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(true);
        exp = "[[ 2 + 4.25i  0   9.4i        1.26      ]\n" +
                " [   56.25    0  -1.45i  -3.36 - 84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 5 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(1);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(true);
        exp = "Full Shape: 2x4\n" +
                "[ [               ...                ]\n" +
                " [ 56.25  0  -1.45i  -3.36 - 84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 6 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(3);
        PrintOptions.setCentering(true);
        exp = "Full Shape: 2x4\n" +
                "[[ 2 + 4.25i  0  ...       1.26      ]\n" +
                " [   56.25    0  ...  -3.36 - 84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 6 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(2);
        PrintOptions.setPadding(2);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(2);
        PrintOptions.setCentering(true);
        exp = "Full Shape: 2x4\n" +
                "[[ 2 + 4.25i  ...       1.26      ]\n" +
                " [   56.25    ...  -3.36 - 84.25i ]]";

        assertEquals(exp, A.toString());

        // ------------------------ Sub-case 6 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(2, 4.25), new CNumber(-0.0002345),
                        new CNumber(0, 9.4), new CNumber(1.2598)},
                {new CNumber(56.25, -0.0024), new CNumber(),
                        new CNumber(0, -1.4545), new CNumber(-3.356, -84.2525)}};
        A = new CMatrix(aEntries);
        PrintOptions.setPrecision(3);
        PrintOptions.setPadding(3);
        PrintOptions.setMaxRows(2);
        PrintOptions.setMaxColumns(4);
        PrintOptions.setCentering(false);
        exp = "[[2 + 4.25i        0   9.4i      1.26               ]\n" +
                " [56.25 - 0.002i   0   -1.455i   -3.356 - 84.253i   ]]";

        assertEquals(exp, A.toString());

        // ------------------------ RESET PRINT OPTIONS ------------------------
        PrintOptions.resetAll();
    }
}
