using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebApiDotNet.Controllers;

namespace WebApiDotNet.Tests
{
    [TestClass]
    public class WeatherForecastControllerUnitTest
    {
        private readonly WeatherForecastController _controller;

        public WeatherForecastControllerUnitTest()
        {
            _controller = new WeatherForecastController(null);
        }

        [TestMethod]
        public void ShouldBeArray()
        {
            Assert.IsTrue(_controller.Get().GetType().IsArray, "Result should be an array");
        }
    }
}
