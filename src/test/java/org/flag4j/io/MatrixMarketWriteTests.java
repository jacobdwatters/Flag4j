package org.flag4j.io;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixMarketWriteTests {
    final static MatrixMarketReader reader = new MatrixMarketReader();

    @Test
    void writeRealDenseTestCase() throws IOException {
        String fileName;
        double[][] expData;
        Matrix exp;
        MatrixMarketReader reader = new MatrixMarketReader();

        // -------------------- sub-case 1  --------------------
        fileName = "./src/test/data/matrix_market/array/ReDe3x3_out.mtx";
        expData = new double[][] {
                {1, 2, 3},
                {4.014e-9, 11512.234, 0.000914},
                {14E56, -31.32, -78e-6}};
        exp = new Matrix(expData);
        MatrixMarketWriter.write(fileName, exp);
        assertEquals(exp, reader.read(fileName));

        // -------------------- sub-case 2  --------------------
        fileName = "./src/test/data/matrix_market/array/ReDe5x2_out.mtx";
        expData = new double[][] {
                {42.88975397315653, -7.229462112800718},
                {77.77634809247053, -72.67239859214298},
                {-28.896793915896055, 58.541359482462894},
                {2.4693564393389664, 66.05800124106074},
                {94.83243949848998, -25.279053739157845}};
        exp = new Matrix(expData);
        MatrixMarketWriter.write(fileName, exp, "generated by Flag4j", "test comment");
        assertEquals(exp, reader.read(fileName));

        // -------------------- sub-case 3  --------------------
        fileName = "./src/test/data/matrix_market/array/ReDe7x14_out.mtx";
        expData = new double[][]{
                {-7.447708816878972, 66.49760248549055, -72.78622988809545, 41.6235915295199, -31.403389317838545, -40.887147695530416, -67.0474509122703, -90.8582545953998, -59.95781767061856, 31.374756851538848, 45.93517429414919, -97.11924214391045, 12.910841152880877, -79.98960506527497},
                {-86.21557987388877, -71.4641313232438, -92.95580839246931, -0.3680447478712665, -97.50930761134322, 83.2789175937055, -42.989851445938186, -76.59661024648996, -69.76534583842295, 65.69107858408114, 49.59567825921417, -62.39713193214867, 2.683250182348644, 90.98196785174684},
                {49.03904351423435, -86.33785802772367, -15.805899967134792, 75.36935485411448, 9.130269125962002, -20.124697340853487, -72.26885624364974, 13.09494188351779, -68.44519881812307, -30.478450017531628, 92.06800046120281, 2.675864196433224, -40.67825029136656, -26.625720694197355},
                {-46.47398884676595, 42.71015403568981, -11.128825519269341, -29.330713269456865, 63.46470586331483, -51.7979348652456, -65.51659087042822, -95.22820507720454, -40.48694455393181, 19.075087332249737, 79.903550772859, -22.39050376201679, -91.05300975473477, -6.022438916613069},
                {-29.15855869563711, 0.6195550397762588, -47.05827490293024, 25.232040385677763, 72.12780013594312, -1.7140198889623122, -86.62250138315524, -42.68306230328673, -1.3061953213015443, -23.319364477617427, 62.32644610164198, -7.13647848547248, -73.07440996183917, 13.485255789617327},
                {43.76237863765289, -13.365730589377776, 82.7804044267719, 56.92302799125429, -21.43440226221219, 24.191485579580103, 70.28914492993198, 32.52871773840192, -37.85533046022198, 43.110571849373144, 42.937881425224106, 43.62166633527525, -47.14035249643109, 21.445940867297807},
                {38.930486250578724, -40.75289725874711, 54.23704681848716, -18.297334314575636, -80.23103566382545, -15.91415205256908, -89.96836670580623, -56.57547635676507, 57.570203890653715, 87.3908251716388, -70.70079949360459, -41.973445835293546, 8.523287416654242, 43.66457374723146}};
        exp = new Matrix(expData);
        MatrixMarketWriter.write(fileName, exp);
        assertEquals(exp, reader.read(fileName));
    }


    @Test
    void writeRealCooTestCase() throws IOException {
        String fileName;
        CooMatrix exp;
        CooMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/data/matrix_market/coordinate/ReCoo5x5_8_out.mtx";
        exp = MatrixMarketMatrices.ReCoo5x5_8;
        MatrixMarketWriter.write(fileName, exp);
        assertEquals(exp, reader.read(fileName));

        // ---------------- sub-case 2 ----------------
        fileName = "./src/test/data/matrix_market/coordinate/jgl009_out.mtx";
        exp = MatrixMarketMatrices.jgl009;
        MatrixMarketWriter.write(fileName, exp);
        assertEquals(exp, reader.read(fileName));

        // ---------------- sub-case 3 ----------------
        fileName = "./src/test/data/matrix_market/coordinate/fidapm05_out.mtx";
        exp = MatrixMarketMatrices.fidapm05;
        MatrixMarketWriter.write(fileName, exp);
        assertEquals(exp, reader.read(fileName));
    }

    @Test
    void writeCooSymmPatternTestCase() throws IOException {
        String fileName;
        CooMatrix exp;
        CooMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/data/matrix_market/coordinate/can___24_out.mtx";
        exp = MatrixMarketMatrices.can___24;
        MatrixMarketWriter.write(fileName, exp, true, "Generated by Flag4j");
        assertEquals(exp, reader.read(fileName));
    }
}
