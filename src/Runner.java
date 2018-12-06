public class Runner {

    public static void main(String[] args) throws Exception {
        if(args.length == 0) {
            System.out.println("Excepted one argument: convert OR predict!");
            return;
        }

        String arg = args[0];
        if(arg.equals("convert")) {
            Main.convertAndProcessAllData();
        } else if(arg.equals("predict")) {
            Predictor.trainAndPredictWithAndWithoutNames();;
        }
    }

}
